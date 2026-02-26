import numpy as np
import torch
import math
import logging
import os
import time
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
from .entity import EntityBelief, init_single_creature, WorldModel
from .physics import PhysicsEngine
from config import Config

# Setup logging to file
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'  # Overwrite log file on each run
)
logger = logging.getLogger(__name__)

# Precompute constants to avoid repeated computation in hot loops
LOG_2PI = math.log(2.0 * math.pi)


# ===== MULTI-THREADING POOL FOR PHYSICS ======
# Auto-detect optimal thread count (avoid oversubscription)
def _get_physics_thread_count():
    """Determine optimal number of worker threads for physics."""
    if Config.NUM_PHYSICS_THREADS is not None:
        return Config.NUM_PHYSICS_THREADS
    # Use all CPUs except 1 (reserved for main thread)
    return max(1, os.cpu_count() - 1)

physics_thread_pool = ThreadPoolExecutor(
    max_workers=_get_physics_thread_count(),
    thread_name_prefix="physics_worker"
)

def compute_reward_parallel(creatures, motor_torques_list, physics_engine, goal_pos=None):
    """
    Compute rewards for multiple creatures in parallel using ThreadPoolExecutor.

    Per-creature goal: each creature carries its own goal_pos so that
    different environments can pursue independent targets simultaneously.
    The shared `goal_pos` argument is used only as a fallback.

    Example:
        # Each creature has been given its own goal via spawn_random_goal(env_id)
        results = compute_reward_parallel(creatures, torques, physics, None)
    """
    def compute_single(creature, motor_torques):
        # Per-creature goal (set by spawn_random_goal).  Falls back to the
        # shared goal_pos for old code paths that haven't migrated yet.
        g = (creature.goal_pos
             if (hasattr(creature, 'goal_pos') and creature.goal_pos is not None)
             else goal_pos)
        reward, com_dist, _ = physics_engine._compute_reward(creature, motor_torques, g)
        return reward, com_dist
    
    # Submit all tasks to thread pool
    futures = [
        physics_thread_pool.submit(compute_single, creature, torques)
        for creature, torques in zip(creatures, motor_torques_list)
    ]
    
    # Collect results
    results = [f.result() for f in futures]
    return results


def detect_available_envs():
    """
    Auto-detect optimal number of parallel environments based on realistic memory model.
    Uses Config.auto_num_envs() which calculates based on observation size and rollout length.
    """
    num_envs = Config.auto_num_envs()
    
    try:
        # Log memory breakdown for debugging
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                gpu_mem_free = sum(gpu.memoryFree for gpu in gpus) if gpus else 0  # MB
            except:
                gpu_mem_free = 0
        else:
            gpu_mem_free = 0
        
        mem = psutil.virtual_memory()
        cpu_mem_free = mem.available / (1024 * 1024)  # Convert to MB
        total_mem_free = gpu_mem_free + cpu_mem_free
        per_env_cost = (Config.ROLLOUT_STEPS * Config.PER_STEP_BYTES) / (1024 * 1024)
        
        logger.info(f"[MEMORY MODEL] GPU={gpu_mem_free:.0f}MB, CPU={cpu_mem_free:.0f}MB, "
                    f"Total={total_mem_free:.0f}MB | Per-env cost={per_env_cost:.1f}MB | "
                    f"Selected {num_envs} environments (budget: {Config.MAX_DATA_THRESHOLD_MB}MB)")
    except Exception as e:
        logger.warning(f"Failed to log memory details: {e}")
    
    return num_envs


class Environment:
    """Manages world state, creatures, goals, and multi-environment support."""
    
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        
        # World dimensions
        self.w = Config.DIM[0]  # width (x)
        self.h = Config.DIM[1]  # height (y)
        self.d = Config.DIM[2]  # depth (z)
        
        # Boundaries
        self.boundary_min = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.boundary_max = torch.tensor([self.w - 1.0, self.h - 1.0, self.d - 1.0], device=device, dtype=dtype)
        
        # World scale for normalization
        self.world_scale_t = torch.tensor([float(self.w), float(self.h), float(self.d)], device=device, dtype=dtype)
        
        # Goal position (single source of truth on device).
        # z is clamped to BODY_INITIAL_HEIGHT — goals are always on the ground plane.
        self.goal_pos_t = torch.tensor([float(self.w - 5), float(self.h - 5), float(Config.BODY_INITIAL_HEIGHT)], device=device, dtype=dtype)
        
        # Multi-environment support
        self.use_vectorized = True
        self.num_envs = Config.NUM_ENVS if Config.NUM_ENVS is not None else detect_available_envs()

        logger.info(f"Vectorized environment enabled with {self.num_envs} parallel environments")
        
        # Creature management
        self.creatures = []  # List[Creature] one per environment
        self.edge_indices = []  # Graph structure per environment
        self.prev_distances = {}  # Distance tracking per environment
        
        self.creature_speed = {}
        for en_id, config in Config.ENTITY_TYPES.items():
            self.creature_speed[en_id] = config.get("speed", 1.0)
    
    def init_creatures(self, model):
        """Initialize creatures for all environments."""
        for env_id in range(self.num_envs):
            creature, edge_idx = init_single_creature(
                model,
                en_id=1,
                pos=(10.0, 10.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                device=self.device
            )
            self.creatures.append(creature)
            self.edge_indices.append(edge_idx)
            self.prev_distances[env_id] = self._distance_to_goal(creature)
    
    def _distance_to_goal(self, creature):
        """Calculate Euclidean distance from creature to its goal (x-y plane)."""
        goal = (
            creature.goal_pos
            if (hasattr(creature, 'goal_pos') and creature.goal_pos is not None)
            else self.goal_pos_t
        )
        dx = creature.pos[0] - goal[0]
        dy = creature.pos[1] - goal[1]
        return torch.sqrt(dx**2 + dy**2)
    
    def spawn_random_goal(self, env_id=0):
        """
        Spawn a new goal for environment `env_id` only.

        Per-creature goals fix the old bug where any env reaching its goal
        would overwrite the single shared goal_pos_t for ALL environments,
        giving every other env a new — wrong — target mid-episode.

        Example:
            # env 3 reaches goal → only its goal changes, others unaffected
            self.spawn_random_goal(env_id=3)
        """
        x_margin = min(20, self.w // 4)
        y_margin = min(20, self.h // 4)
        x = np.random.uniform(x_margin, self.w - x_margin)
        y = np.random.uniform(y_margin, self.h - y_margin)
        # Goals are on the ground plane — z=BODY_INITIAL_HEIGHT (~0.3m standing height).
        # z is the UP axis in this engine (gravity = -z, ground at z=0).
        # Aerial goals (old: z ∈ [0, d-5] = [0, 59]) are unreachable by ground robots.
        z = float(Config.BODY_INITIAL_HEIGHT)
        goal = torch.tensor([float(x), float(y), float(z)],
                            device=self.device, dtype=self.dtype)

        # Per-creature goal (independent per env)
        if env_id < len(self.creatures):
            self.creatures[env_id].goal_pos = goal.clone()

        # Keep shared goal_pos_t in sync so the Renderer (env 0 view) is correct
        self.goal_pos_t[0] = float(x)
        self.goal_pos_t[1] = float(y)
        self.goal_pos_t[2] = float(z)
        if env_id < len(self.creatures):
            self.prev_distances[env_id] = self._distance_to_goal(self.creatures[env_id])
    
    def observe(self, creature):
        """
        Observation for quadruped in AGENT-CENTERED WORLD (34D).

        Uses creature.goal_pos (per-creature goal) so two environments
        can have different goals at the same time.  Falls back to the
        shared self.goal_pos_t if goal_pos is not yet initialised.

        Returns: (1, 34) tensor
            [joint_angles(12), joint_vels(12), foot_contact(4),
             orientation(3), goal_relative(3)]
        """
        goal = (
            creature.goal_pos
            if (hasattr(creature, 'goal_pos') and creature.goal_pos is not None)
            else self.goal_pos_t
        )
        obs_list = [
            creature.joint_angles,      # (12,)
            creature.joint_velocities,  # (12,)
            creature.foot_contact,      # (4,)
            creature.orientation,       # (3,)
            goal - creature.pos,        # (3,) goal in agent-centred frame
        ]
        obs = torch.cat(obs_list).unsqueeze(0).to(self.device)  # (1, 34)
        return obs


class TrainingEngine:
    """Manages RL training: data collection, PPO, world model."""
    
    def __init__(self, device, dtype, environment, physics_engine):
        self.device = device
        self.dtype = dtype
        self.env = environment
        self.physics = physics_engine
        
        # RL models - QUADRUPED (34D observation, 12D motor torques)
        # WARNING: OBS_DIM=34 and ACTION_DIM=12 are hardcoded; changes affect multiple modules
        NN_OBS_DIM = 34
        NN_NUM_ACTIONS = 12
        
        self.model = EntityBelief(
            obs_dim=NN_OBS_DIM,
            embed_dim=Config.EMBED_DIM,
            gat_out_dim=Config.GAT_OUT_DIM,
            gat_heads=Config.GAT_HEADS,
            lstm_hidden=Config.LSTM_HIDDEN,
            num_actions=NN_NUM_ACTIONS,
        ).to(device)

        self.num_actions = NN_NUM_ACTIONS

        # Split compute across two GPUs when available:
        #   GPU 0 (self.device):    policy inference + PPO training (EntityBelief)
        #   GPU 1 (self.wm_device): world model training (encoder + dynamics heads)
        # DataParallel on the tiny EntityBelief (obs=34 → hidden=256) is NOT used;
        # its per-step forward pass is too small to amortise DP synchronisation cost.
        self.wm_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
        if self.wm_device != device:
            logger.info("[Multi-GPU] EntityBelief on cuda:0, WorldModel on cuda:1")
            print("[Multi-GPU] EntityBelief on cuda:0, WorldModel on cuda:1")

        # PATTERN A: WorldModel owns its own encoder on wm_device.
        # After each world model update, encoder weights are synced back to the
        # policy's frozen encoder via cross-device copy (see train_world_model).
        self.world_model = WorldModel(
            obs_dim=NN_OBS_DIM,
            action_dim=NN_NUM_ACTIONS,
            latent_dim=Config.EMBED_DIM,
            hidden_dim=Config.WORLD_MODEL_HIDDEN_DIM,
            encoder=None,  # own encoder on wm_device, synced back after each update
        ).to(self.wm_device)

        # Optimizers
        # world_model_optimizer owns the full world model (including its encoder)
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=Config.WORLD_MODEL_LR,
            weight_decay=Config.WORLD_MODEL_WEIGHT_DECAY
        )
        
        # PPO optimizer only touches policy parameters (encoder is frozen in EntityBelief)
        ppo_params = [p for p in self.model.policy.parameters()]
        self.optimizer = torch.optim.Adam(ppo_params, lr=Config.LR)
        
        # Training hyperparameters
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        self.entropy_coef = Config.ENTROPY_COEF
        self.value_coef = Config.VALUE_COEF
        
        # Action scaling
        self.action_scale = torch.tensor([1.0, 1.0, Config.ACTION_SCALE_Z], device=device, dtype=dtype)
        
        # Numerical stability for log probability computation
        self.log_eps = 1e-6
        
        # Checkpoint
        Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Config.CHECKPOINT_DIR / "model.pt"
        
        # Tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.frames_per_step = Config.TARGET_FPS // Config.EVAL_STEPS_PER_SEC
    
    def freeze_encoder(self):
        """Freeze shared encoder for PPO training (Dreamer not active)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze shared encoder for Dreamer training on replay buffer."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
    
    def save_checkpoint(self):
        """Save model state to checkpoint file."""
        try:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'world_model_state': self.world_model.state_dict(),
                'world_model_optimizer_state': self.world_model_optimizer.state_dict(),
                'episode_count': self.episode_count,
                'episode_rewards': self.episode_rewards,
            }
            torch.save(checkpoint, self.checkpoint_path)
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _run_model(self, obs, edge_idx, prev_state):
        """Policy forward pass – always on self.device (cuda:0)."""
        return self.model(obs, edge_idx, prev_state=prev_state)

    def load_checkpoint(self):
        """Load model state from checkpoint if available."""
        if not Config.LOAD_CHECKPOINT:
            logger.info("LOAD_CHECKPOINT=False, starting fresh training")
            return
        
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.world_model.load_state_dict(checkpoint['world_model_state'])
                self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer_state'])
                self.episode_count = checkpoint['episode_count']
                self.episode_rewards = checkpoint['episode_rewards']
                logger.info(f"Checkpoint loaded (episode {self.episode_count})")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        else:
            logger.info("No checkpoint found, starting fresh training")
    
    def collect_trajectories_vectorized(self, max_steps=500):
        """
        Collect trajectories from ALL environments in parallel (batched).
        Maintains separate RNN states, goals, and creatures per environment.
        
        Returns:
            list of trajectories, one per environment
        """
        self.model.eval()
        
        num_envs = self.env.num_envs
        
        # Initialize trajectory list for each environment
        trajectories = [[] for _ in range(num_envs)]
        step_counts = [0] * num_envs
        
        # Reset all creatures
        for env_id in range(num_envs):
            creature = self.env.creatures[env_id]
            creature.pos = torch.tensor([0.0, 0.0, 0.5], dtype=self.dtype, device=self.device)
            creature.velocity = torch.zeros(3, dtype=self.dtype, device=self.device)
            creature.orientation = torch.zeros(3, dtype=self.dtype, device=self.device)
            creature.joint_angles = torch.tensor([0.3, 0.6, 0.3] * 4, dtype=self.dtype, device=self.device)
            creature.joint_velocities = torch.zeros(12, dtype=self.dtype, device=self.device)
            creature.foot_contact = torch.ones(4, dtype=self.dtype, device=self.device)
            h0, c0 = self.model.init_state(1, self.device, self.dtype)
            creature.rnn_state = (h0, c0)
            self.env.spawn_random_goal(env_id)
            self.physics.reset_body_state(creature)  # sync GPU body state after reset
        
        # Main loop: run all environments until they all reach max_steps
        for global_step in range(max_steps):
            # Collect observations from all environments
            obs_list = []
            for env_id in range(num_envs):
                if step_counts[env_id] < max_steps:
                    creature = self.env.creatures[env_id]
                    obs = self.env.observe(creature)  # (1, 34)
                    obs_list.append(obs)
            
            if not obs_list:
                break  # All environments done
            
            # Stack into batch: (batch_size, 34)
            obs_batch = torch.cat(obs_list, dim=0)  # (num_active_envs, 34)
            
            # Get actions from model for all environments
            with torch.no_grad():
                # NOTE: This assumes model can handle variable batch sizes
                # If model doesn't support this, we may need to pad to fixed size
                try:
                    # Try batch processing
                    edge_idx = self.env.edge_indices[0]  # Use first env's edge index (should be same for all)
                    
                    # Get RNN states for active environments
                    h_list = []
                    c_list = []
                    env_id_active = 0
                    for env_id in range(num_envs):
                        if step_counts[env_id] < max_steps:
                            creature = self.env.creatures[env_id]
                            h, c = creature.rnn_state
                            h_list.append(h)
                            c_list.append(c)
                    
                    h_batch = torch.cat(h_list, dim=0)  # (num_active, hidden)
                    c_batch = torch.cat(c_list, dim=0)  # (num_active, hidden)
                    
                    (mu, log_std), values, (new_h, new_c) = self._run_model(
                        obs_batch, edge_idx, (h_batch, c_batch)
                    )
                    
                    # Sample actions (Gaussian in R^12, no tanh in policy)
                    std = torch.exp(log_std)
                    actions = mu + torch.randn_like(mu) * std  # (batch, 12)
                    
                    # Compute log probs for Gaussian
                    log_probs = -0.5 * ((actions - mu) ** 2 / (std ** 2)).sum(dim=1)
                    log_probs = log_probs - log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI
                    
                    # Track which actions would be clamped (for diagnostics)
                    actions_clamped = (torch.abs(actions) > 5.0).float()  # (batch, 12)
                    
                    # Collect creatures and actions for parallel reward computation
                    active_creatures = []
                    active_actions = []
                    active_env_ids = []
                    env_id_active = 0
                    
                    for env_id in range(num_envs):
                        if step_counts[env_id] < max_steps:
                            creature = self.env.creatures[env_id]
                            # Gaussian → clamp/scale torques
                            # Actions are in R, physics clamps them to [-MAX_TORQUE, MAX_TORQUE]
                            # We can scale them if needed, but let's assume mu/std learn the range
                            motor_torques = actions[env_id_active] # (12,)
                            active_creatures.append(creature)
                            active_actions.append(motor_torques)
                            active_env_ids.append(env_id)
                            env_id_active += 1
                    
                    # Compute rewards + physics step in one GPU-batched call.
                    # Replaces N sequential CPU _compute_reward calls with a single
                    # step_batch() dispatch that runs joint dynamics, FK, contacts,
                    # rigid-body integration and reward on GPU in parallel.
                    rewards_batch, distances_batch, _ = self.physics.step_batch(
                        active_creatures,
                        torch.stack(active_actions),                         # (N_active, 12)
                        torch.stack([c.goal_pos for c in active_creatures]), # (N_active, 3) per-creature
                    )
                    
                    # Process results for each environment
                    env_id_active = 0
                    for env_idx, (creature, env_id) in enumerate(zip(active_creatures, active_env_ids)):
                        if step_counts[env_id] < max_steps:
                            # Extract results for this environment
                            mu_i = mu[env_id_active:env_id_active+1]
                            action_i = actions[env_id_active:env_id_active+1]
                            value_i = values[env_id_active:env_id_active+1]
                            log_prob_i = log_probs[env_id_active]
                            
                            # Update RNN state
                            creature.rnn_state = (
                                new_h[env_id_active:env_id_active+1],
                                new_c[env_id_active:env_id_active+1]
                            )
                            
                            # Get pre-computed reward and distance from batched step
                            reward   = rewards_batch[env_idx]
                            com_dist = distances_batch[env_idx]
                            
                            # Check if done
                            done = float(com_dist) < self.physics.com_distance_threshold
                            if done:
                                self.env.spawn_random_goal(env_id)
                                # NOTE: creature.rnn_state intentionally NOT reset here
                                # This is a CONTINUING task - the agent remembers across goals
                                # Training will handle episode boundaries separately
                            
                            # Store next_obs (cheap tensor cat, no model forward pass).
                            # next_value will be computed by shifting value[t+1] after
                            # the main loop, with one bootstrap call per env at the end.
                            # This replaces steps × num_envs forward-pass calls with
                            # num_envs calls total.
                            next_obs = self.env.observe(creature)

                            # Store transition
                            trajectories[env_id].append({
                                'obs': obs_list[env_id_active],
                                'action': action_i,
                                'reward': reward,
                                'value': value_i.squeeze().detach(),
                                'next_obs': next_obs,    # for world model training
                                'done': float(done),
                                'old_log_prob': log_prob_i.detach(),
                                'joint_vels': creature.joint_velocities.detach().clone(),
                                'foot_contacts': creature.foot_contact.detach().clone(),
                                'action_clamped': actions_clamped[env_id_active],
                            })
                            
                            step_counts[env_id] += 1
                            env_id_active += 1
                
                except RuntimeError as e:
                    # Fallback: if batch processing fails, fall back to sequential
                    logger.warning(f"Batch processing failed: {e}, falling back to sequential")
                    return self.collect_trajectory_sequential(max_steps)
        
        # --- Bootstrap next_values (one forward pass per env, not per step) ---
        # next_value[t] = value[t+1] for t < T-1  (already in trajectory).
        # For the last step we run one model forward pass to get the bootstrap.
        #
        # Old cost: steps × num_envs separate model calls inside the loop.
        # New cost: num_envs model calls total, once at the end.
        for env_id in range(num_envs):
            traj = trajectories[env_id]
            if not traj:
                continue
            # Shift: each step's next_value is the following step's value
            for t in range(len(traj) - 1):
                traj[t]['next_value'] = traj[t + 1]['value'].clone()
            # Bootstrap last step (one call per env)
            creature = self.env.creatures[env_id]
            with torch.no_grad():
                last_obs = self.env.observe(creature)
                (_, _), bootstrap_val, _ = self._run_model(
                    last_obs,
                    self.env.edge_indices[0],
                    creature.rnn_state,
                )
            traj[-1]['next_value'] = bootstrap_val.squeeze().detach()

        # Wrap trajectories with metadata for batch training
        trajectories_wrapped = []
        for env_id, traj in enumerate(trajectories):
            if traj:  # Only include non-empty trajectories
                trajectories_wrapped.append({
                    'env_id': env_id,
                    'edge_index': self.env.edge_indices[env_id],
                    'trajectory': traj
                })
        return trajectories_wrapped
    
    def collect_trajectory_sequential(self, max_steps=500):
        """
        Sequential trajectory collection (original implementation).
        Used as fallback if vectorized version fails.
        Returns list of dicts: {env_id, edge_index, trajectory_data}
        """
        trajectories = []
        for env_id in range(self.env.num_envs):
            traj = self.collect_trajectory(max_steps=max_steps, env_id=env_id)
            if traj:
                # Wrap trajectory with metadata to prevent index misalignment
                trajectories.append({
                    'env_id': env_id,
                    'edge_index': self.env.edge_indices[env_id],
                    'trajectory': traj
                })
        return trajectories
    
    def collect_trajectory(self, max_steps=500, env_id=0):
        """
        Collect one episode trajectory for quadruped balance task.
        Actions are 12D motor torques for quadruped legs.
        """
        self.model.eval()
        
        trajectory = []
        step_count = 0
        
        creature = self.env.creatures[env_id]
        edge_idx = self.env.edge_indices[env_id]
        
        # Reset creature to standing pose
        creature.pos = torch.tensor([0.0, 0.0, 0.5], dtype=self.dtype, device=self.device)
        creature.velocity = torch.zeros(3, dtype=self.dtype, device=self.device)
        creature.orientation = torch.zeros(3, dtype=self.dtype, device=self.device)
        
        # Reset legs to neutral standing position
        creature.joint_angles = torch.tensor(
            [0.3, 0.6, 0.3] * 4,  # 4 legs
            dtype=self.dtype, device=self.device
        )
        creature.joint_velocities = torch.zeros(12, dtype=self.dtype, device=self.device)
        creature.foot_contact = torch.ones(4, dtype=self.dtype, device=self.device)
        
        # Reset both GPU body-state tensors and CPU RigidBody so that the
        # sequential physics path (_update_joint_dynamics_cpu) starts the new
        # episode from the correct position with zero velocity, not the stale
        # state left over from the end of the previous episode.
        self.physics.reset_body_state(creature)

        # Reset RNN state
        h0, c0 = self.model.init_state(1, self.device, self.dtype)
        creature.rnn_state = (h0, c0)
        
        # Spawn random goal within reachable distance
        self.env.spawn_random_goal(env_id)
        
        while step_count < max_steps:
            # Get observation
            obs = self.env.observe(creature)
            
            # Get action from policy (Gaussian in R^12, no tanh in policy)
            with torch.no_grad():
                (mu, log_std), value, new_state = self._run_model(obs, edge_idx, creature.rnn_state)
            
            creature.rnn_state = new_state
            
            # Sample action (Gaussian)
            std = torch.exp(log_std)
            action = mu + torch.randn_like(mu) * std # (1, 12)
            
            # Compute log probability
            log_prob = -0.5 * ((action - mu) ** 2 / (std ** 2)).sum(dim=1)
            log_prob = log_prob - log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI
            
            # Track which action components exceed the clamp threshold
            action_clamped = (torch.abs(action) > 5.0).float()  # (1, 12)
            
            # Physics clamps the torques anyway
            motor_torques = action[0] # (12,)
            
            # Apply physics and get reward (use per-creature goal)
            goal = (
                creature.goal_pos
                if (hasattr(creature, 'goal_pos') and creature.goal_pos is not None)
                else self.env.goal_pos_t
            )
            reward, com_dist, stability_metrics = self.physics._compute_reward(
                creature, motor_torques, goal
            )
            
            # Check if goal reached (COM within threshold)
            done = float(com_dist) < self.physics.com_distance_threshold
            if done:
                self.env.spawn_random_goal(env_id)
                # NOTE: creature.rnn_state intentionally NOT reset here
                # This is a CONTINUING task - the agent remembers across goals
                # Training will handle episode boundaries separately
            
            # Get next value estimate
            with torch.no_grad():
                next_obs = self.env.observe(creature)
                # RNN state carries forward to next goal for continuity
                (next_mu, next_log_std), next_value, _ = self._run_model(
                    next_obs, edge_idx, creature.rnn_state
                )
            
            # Store transition
            trajectory.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'value': value.squeeze().detach(),
                'next_value': next_value.squeeze().detach(),
                'done': float(done),
                'old_log_prob': log_prob[0].detach(),
                'joint_vels': creature.joint_velocities.detach().clone(),
                'foot_contacts': creature.foot_contact.detach().clone(),
                'action_clamped': action_clamped.squeeze(),  # Track clamp per step
            })
            
            step_count += 1
        
        self.step_count = step_count
        return trajectory
    
    def train_on_trajectory(self, trajectory):
        """Train on a collected trajectory using PPO for quadruped motor control.
        
        PATTERN A ARCHITECTURE:
        1. Unfreeze and train Dreamer (encoder + world model) on replay buffer
        2. Freeze encoder and train PPO policy on on-policy rollouts
        This prevents representation drift from mixed objectives.
        """
        if not trajectory:
            return
        
        # ============ DREAMER TRAINING (off-policy on replay buffer) ============
        # WorldModel trains its own encoder on wm_device and syncs weights back
        # to the frozen policy encoder via cross-device copy in train_world_model.
        wm_loss = self.train_world_model(trajectory)
        logger.info(f"  [Dreamer] World Model Loss: {wm_loss:.4f}")
        
        # ============ PPO TRAINING (on-policy with frozen encoder) ============
        
        self.model.train()
        
        # Compute advantages
        rewards = torch.stack([t['reward'] for t in trajectory]).squeeze(-1)
        values = torch.stack([t['value'] for t in trajectory])
        next_values = torch.stack([t['next_value'] for t in trajectory])
        dones = torch.tensor([t['done'] for t in trajectory], dtype=torch.float32, device=self.device)
        old_log_probs = torch.cat([t['old_log_prob'].unsqueeze(0) for t in trajectory])
        
        deltas = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        
        advantages_raw = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(trajectory))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages_raw[t] = gae
        
        returns = advantages_raw + values.squeeze()
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        
        obs_list = [t['obs'] for t in trajectory]
        action_list = [t['action'] for t in trajectory]
        
        for _ in range(Config.PPO_EPOCHS):
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                # forward_sequence batches the encoder over all T observations at
                # once instead of T separate (1, obs_dim) calls.
                # Example T=1024, 4 epochs: was 4096 encoder calls → now 4.
                # The LSTMCell unroll stays sequential for done-boundary masking.
                obs_seq = torch.cat(obs_list, dim=0)    # (T, obs_dim)
                (mu_seq, log_std_seq), value_seq, _ = self.model.forward_sequence(
                    obs_seq, dones=dones
                )

                action_batch = torch.cat(action_list, dim=0)  # (T, 12)
                std = torch.exp(log_std_seq)
                
                # Log probability for 12D Gaussian policy (no tanh)
                log_prob_seq = -0.5 * ((action_batch - mu_seq) ** 2 / (std ** 2)).sum(dim=1)
                log_prob_seq = log_prob_seq - log_std_seq.sum(dim=1) - 0.5 * 12 * LOG_2PI

                # log_ratio must stay outside no_grad so that ratio carries
                # a gradient back through log_prob_seq → mu_seq → policy_mu.
                log_ratio = log_prob_seq - old_log_probs
                with torch.no_grad():
                    kl = ((torch.exp(log_ratio.detach()) - 1) - log_ratio.detach()).mean()

                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - Config.PPO_CLIP_RATIO, 1.0 + Config.PPO_CLIP_RATIO)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                clip_frac = (torch.abs(ratio - 1.0) > Config.PPO_CLIP_RATIO).float().mean()
                
                v_pred = value_seq.squeeze()
                v_old = values.squeeze().detach()
                v_clipped = v_old + torch.clamp(v_pred - v_old, -0.2, 0.2)
                value_loss = torch.max((v_pred - returns)**2, (v_clipped - returns)**2).mean()
                
                entropy = (0.5 * (1.0 + LOG_2PI) + log_std_seq).sum(dim=1).mean()
                
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        # Diagnostics
        with torch.no_grad():
            joint_vels = torch.stack([t['joint_vels'] for t in trajectory]) # (T, 12)
            sat_rate = (torch.abs(joint_vels) >= 10.0).float().mean()
            
            contacts = torch.stack([t['foot_contacts'] for t in trajectory]) # (T, 4)
            mean_contacts = contacts.mean(dim=0)
            all_4_freq = (contacts.sum(dim=1) == 4.0).float().mean()
            
            action_clamp = torch.stack([t['action_clamped'] for t in trajectory])  # (T, 12)
            clamp_frac = action_clamp.mean().item()
            
            action_mean = action_batch.mean(dim=0)
            action_std = action_batch.std(dim=0)

        episode_reward = sum(float(t['reward']) for t in trajectory)
        self.episode_rewards.append(episode_reward)
        msg = f"Ep {self.episode_count}: reward={episode_reward:.2f}, policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, entropy={entropy.item():.4f}, KL={kl.item():.4f}, clip={clip_frac.item():.4f}, action_clamp={clamp_frac:.2%}"
        logger.info(msg)
        logger.info(f"  Actions: mean={action_mean.mean().item():.3f}, std={action_std.mean().item():.3f} | Sat Rate: {sat_rate.item():.2%}")
        logger.info(f"  Contacts: {mean_contacts.tolist()} | All 4 Freq: {all_4_freq.item():.2%}")
        print(msg)
    
    def train_on_trajectories_batch(self, trajectories_wrapped):
        """Train on multiple trajectories from vectorized environments.
        
        PATTERN A ARCHITECTURE:
        1. Unfreeze and train Dreamer (encoder + world model) on replay buffer
        2. Freeze encoder and train PPO policy on on-policy rollouts
        This prevents representation drift from mixed objectives.
        
        Args:
            trajectories_wrapped: list of dicts with {'env_id', 'edge_index', 'trajectory'}
        """
        if not trajectories_wrapped:
            return
        
        # Filter out empty trajectories, preserving metadata
        trajectories_wrapped = [t for t in trajectories_wrapped if len(t['trajectory']) > 0]
        if not trajectories_wrapped:
            logger.warning("No valid trajectories to train on")
            return
        
        logger.info(f"[BATCH DEBUG] Training on {len(trajectories_wrapped)} trajectories")
        for i, traj_data in enumerate(trajectories_wrapped):
            traj = traj_data['trajectory']
            traj_reward = sum(float(step['reward']) for step in traj)
            logger.info(f"  Traj {i} (env_id={traj_data['env_id']}): len={len(traj)}, total_reward={traj_reward:.2f}")
        
        # ============ DREAMER TRAINING (off-policy on replay buffer) ============
        # WorldModel trains its own encoder on wm_device; weights synced in train_world_model.
        total_wm_loss = 0.0
        for traj_data in trajectories_wrapped:
            wm_loss = self.train_world_model(traj_data['trajectory'])
            total_wm_loss += wm_loss
        avg_wm_loss = total_wm_loss / len(trajectories_wrapped)
        logger.info(f"  [Dreamer] Batch World Model Loss: {avg_wm_loss:.4f} ({len(trajectories_wrapped)} envs)")
        
        # ============ PPO TRAINING (on-policy with frozen encoder) ============
        
        self.model.train()
        
        all_rewards = []
        all_values = []
        all_next_values = []
        all_dones = []
        all_old_log_probs = []
        all_obs = []
        all_actions = []
        all_joint_vels = []
        all_foot_contacts = []
        all_action_clamps = []
        
        for traj_data in trajectories_wrapped:
            trajectory = traj_data['trajectory']
            edge_idx = traj_data['edge_index']
            
            rewards = torch.stack([t['reward'] for t in trajectory]).squeeze(-1)
            values = torch.stack([t['value'] for t in trajectory])
            next_values = torch.stack([t['next_value'] for t in trajectory])
            dones = torch.tensor([t['done'] for t in trajectory], dtype=torch.float32, device=self.device)
            old_log_probs = torch.cat([t['old_log_prob'].unsqueeze(0) for t in trajectory])
            obs_list = [t['obs'] for t in trajectory]
            action_list = [t['action'] for t in trajectory]
            joint_vels = torch.stack([t['joint_vels'] for t in trajectory])
            foot_contacts = torch.stack([t['foot_contacts'] for t in trajectory])
            action_clamps = torch.stack([t['action_clamped'] for t in trajectory])
            
            all_rewards.append(rewards)
            all_values.append(values)
            all_next_values.append(next_values)
            all_dones.append(dones)
            all_old_log_probs.append(old_log_probs)
            all_obs.append(obs_list)
            all_actions.append(action_list)
            all_joint_vels.append(joint_vels)
            all_foot_contacts.append(foot_contacts)
            all_action_clamps.append(action_clamps)
        
        rewards_batch = torch.cat(all_rewards)
        values_batch = torch.cat(all_values)
        next_values_batch = torch.cat(all_next_values)
        dones_batch = torch.cat(all_dones)
        old_log_probs_batch = torch.cat(all_old_log_probs)
        
        # Compute GAE independently per trajectory to prevent advantage
        # contamination across environment boundaries.  A single reversed loop
        # over the concatenated buffer would let the GAE accumulated from
        # env N+1 bleed backwards into the last steps of env N.
        all_advantages_raw = []
        all_returns = []
        for r_t, v_t, nv_t, d_t in zip(all_rewards, all_values, all_next_values, all_dones):
            delta_t = r_t + self.gamma * nv_t.squeeze() * (1 - d_t) - v_t.squeeze()
            adv_t = torch.zeros_like(r_t)
            gae = 0
            for t in reversed(range(len(r_t))):
                gae = delta_t[t] + self.gamma * self.gae_lambda * (1 - d_t[t]) * gae
                adv_t[t] = gae
            all_advantages_raw.append(adv_t)
            all_returns.append(adv_t + v_t.squeeze())

        advantages_raw = torch.cat(all_advantages_raw)
        returns_batch  = torch.cat(all_returns)
        advantages_batch = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        # Precompute obs tensors and action_batch once – observations and actions
        # are fixed across PPO epochs; only the model forward pass changes.
        obs_seqs = []   # one (T, obs_dim) tensor per trajectory
        traj_lengths = []
        step_offset = 0
        for obs_list_t in all_obs:
            T_traj = len(obs_list_t)
            obs_seqs.append(torch.cat(obs_list_t, dim=0))  # (T, obs_dim)
            traj_lengths.append(T_traj)
            step_offset += T_traj

        all_actions_flat = []
        for action_list in all_actions:
            all_actions_flat.extend(action_list)
        action_batch = torch.cat(all_actions_flat, dim=0)  # (total_T, 12)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for epoch in range(Config.PPO_EPOCHS):
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                # Batch encoder across all T steps per trajectory (one big matmul)
                # instead of T × num_envs × PPO_EPOCHS separate (1, obs) calls.
                # E.g. T=512, 32 envs, 10 epochs: was 163 840 calls → now 320.
                all_mus, all_log_stds, all_vals_epoch = [], [], []
                step_offset = 0
                for obs_seq, T_traj in zip(obs_seqs, traj_lengths):
                    t_dones = dones_batch[step_offset:step_offset + T_traj]
                    (mu_t, log_std_t), val_t, _ = self.model.forward_sequence(
                        obs_seq, dones=t_dones
                    )
                    all_mus.append(mu_t)
                    all_log_stds.append(log_std_t)
                    all_vals_epoch.append(val_t)
                    step_offset += T_traj

                mu_seq      = torch.cat(all_mus, dim=0)
                log_std_seq = torch.cat(all_log_stds, dim=0)
                value_seq   = torch.cat(all_vals_epoch, dim=0)

                mu = mu_seq
                std = torch.exp(log_std_seq)
                
                # Log probability for 12D Gaussian policy (no tanh)
                log_prob_seq = -0.5 * ((action_batch - mu) ** 2 / (std ** 2)).sum(dim=1)
                log_prob_seq = log_prob_seq - log_std_seq.sum(dim=1) - 0.5 * 12 * LOG_2PI

                # log_ratio must stay outside no_grad so that ratio carries
                # a gradient back through log_prob_seq → mu_seq → policy_mu.
                log_ratio = log_prob_seq - old_log_probs_batch
                with torch.no_grad():
                    kl = ((torch.exp(log_ratio.detach()) - 1) - log_ratio.detach()).mean()

                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - Config.PPO_CLIP_RATIO, 1.0 + Config.PPO_CLIP_RATIO)
                policy_loss = -torch.min(ratio * advantages_batch, clipped_ratio * advantages_batch).mean()
                
                clip_frac = (torch.abs(ratio - 1.0) > Config.PPO_CLIP_RATIO).float().mean()
                
                v_pred = value_seq.squeeze()
                v_old = values_batch.squeeze().detach()
                v_clipped = v_old + torch.clamp(v_pred - v_old, -0.2, 0.2)
                value_loss = torch.max((v_pred - returns_batch)**2, (v_clipped - returns_batch)**2).mean()
                
                entropy = (0.5 * (1.0 + LOG_2PI) + log_std_seq).sum(dim=1).mean()
                
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Diagnostics
        with torch.no_grad():
            joint_vels_batch = torch.cat(all_joint_vels)
            sat_rate = (torch.abs(joint_vels_batch) >= 10.0).float().mean()
            
            contacts_batch = torch.cat(all_foot_contacts)
            mean_contacts = contacts_batch.mean(dim=0)
            all_4_freq = (contacts_batch.sum(dim=1) == 4.0).float().mean()
            
            action_clamps_batch = torch.cat(all_action_clamps)
            clamp_frac = action_clamps_batch.mean().item()
            
            action_mean_per_joint = action_batch.mean(dim=0)
            action_std_per_joint = action_batch.std(dim=0)
        
        # Calculate average reward across all trajectories
        batch_reward = sum(sum(float(t['reward']) for t in traj_data['trajectory']) for traj_data in trajectories_wrapped) / len(trajectories_wrapped)
        self.episode_rewards.append(batch_reward)
        
        # Log debugging info
        msg = f"Batch Ep {self.episode_count}: avg_reward={batch_reward:.2f}, policy_loss={total_policy_loss/Config.PPO_EPOCHS:.4f}, value_loss={total_value_loss/Config.PPO_EPOCHS:.4f}, entropy={entropy.item():.4f}, KL={kl.item():.4f}, clip={clip_frac.item():.4f}, action_clamp={clamp_frac:.2%}"
        logger.info(msg)
        logger.info(f"  Actions: mean={action_mean_per_joint.mean().item():.3f}, std={action_std_per_joint.mean().item():.3f} | Sat Rate: {sat_rate.item():.2%}")
        logger.info(f"  Contacts: {mean_contacts.tolist()} | All4 Freq: {all_4_freq.item():.2%}")
        print(msg)
    
    def train_world_model(self, trajectory):
        """Train world model on trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        self.world_model.train()
        
        total_loss = 0.0
        num_batches = max(1, len(trajectory) // 32)
        
        for step in range(num_batches):
            idx = torch.randint(0, len(trajectory), (32,))
            
            batch_obs = []
            batch_actions = []
            batch_next_obs = []
            batch_rewards = []
            
            for i in idx.tolist():
                batch_obs.append(trajectory[i]['obs'])
                batch_actions.append(trajectory[i]['action'])
                # Use stored next_obs (set during collection) so that done-boundary
                # transitions are correctly represented.  Falls back to the next
                # step's obs for legacy trajectories that pre-date this fix.
                nxt = trajectory[i].get('next_obs',
                                         trajectory[min(i + 1, len(trajectory) - 1)]['obs'])
                batch_next_obs.append(nxt)
                batch_rewards.append(trajectory[i]['reward'].unsqueeze(0))
            
            obs_batch      = torch.cat(batch_obs,      dim=0).to(self.wm_device)
            actions_batch  = torch.cat(batch_actions,  dim=0).to(self.wm_device)
            # next_obs_batch: the reconstruction target for decode(next_latent).
            # batch_next_obs was collected but never materialised into a tensor —
            # using obs_batch as the target would train the decoder to map the
            # *next* latent back to the *current* observation, a contradictory signal.
            next_obs_batch = torch.cat(batch_next_obs, dim=0).to(self.wm_device)
            rewards_batch  = torch.cat(batch_rewards,  dim=0).squeeze(-1).to(self.wm_device)
            
            next_latent, pred_rewards, pred_dones, recon_obs = self.world_model(obs_batch, actions_batch)
            
            # recon_obs = decode(next_latent) → should match next observation
            recon_loss = torch.nn.functional.mse_loss(recon_obs, next_obs_batch)
            reward_loss = torch.nn.functional.mse_loss(pred_rewards.squeeze(-1), rewards_batch)
            wm_loss = recon_loss + reward_loss
            
            self.world_model_optimizer.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_model_optimizer.step()
            # Sync world-model encoder (wm_device) → policy encoder (device).
            # Keeps the frozen policy encoder up to date with representation learning.
            if self.wm_device != self.device:
                with torch.no_grad():
                    for p_src, p_dst in zip(self.world_model.encoder.parameters(),
                                             self.model.encoder.parameters()):
                        p_dst.data.copy_(p_src.data)

            total_loss += wm_loss.detach().item()
        
        return total_loss / num_batches


class System:
    """Lightweight orchestrator that coordinates all components."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # Eagerly initialize CUDA contexts on all visible GPUs before any model
        # or cuBLAS work.  Without this PyTorch lazily creates the context on the
        # first op, which causes the "no current CUDA context" / cuBLAS warning
        # when DataParallel dispatches to GPU 1 before GPU 0's context exists.
        if self.device.type == "cuda":
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.init()
                    torch.zeros(1, device=f"cuda:{i}")  # forces context creation

        # Initialize components
        self.env = Environment(self.device, self.dtype)
        self.physics = PhysicsEngine(self.device, self.dtype, self.env)
        self.training = TrainingEngine(self.device, self.dtype, self.env, self.physics)
        
        # Initialize creatures with trained model
        self.env.init_creatures(self.training.model)
        self.edge_index = self.env.edge_indices[0]  # For compatibility
    
    def main(self):
        """Main training loop orchestrator."""
        try:
            self.training.load_checkpoint()
            
            msg = "=" * 60
            logger.info(msg)
            print(msg)
            msg = f"TRAINING PHASE - {self.env.num_envs} parallel env(s)"
            logger.info(msg)
            print(msg)
            msg = "=" * 60
            logger.info(msg)
            print(msg)
            
            training_complete = False
            trajectories_batch = []
            
            while not training_complete:
                ep_start = time.time()

                if self.env.use_vectorized:
                    # Parallel trajectory collection across all environments
                    trajectories_batch = self.training.collect_trajectories_vectorized(max_steps=Config.MAX_STEPS_PER_EPISODE)
                    logger.info(f"Collected {len(trajectories_batch)} trajectories in parallel")
                    
                    # Train once per batch
                    if trajectories_batch and any(trajectories_batch):
                        self.training.train_on_trajectories_batch(trajectories_batch)
                        self.training.episode_count += 1
                        self.training.save_checkpoint()
                else:
                    # Single environment
                    trajectory = self.training.collect_trajectory(max_steps=Config.MAX_STEPS_PER_EPISODE, env_id=0)
                    
                    if trajectory:
                        self.training.train_on_trajectory(trajectory)
                        self.training.episode_count += 1
                        self.training.save_checkpoint()
                
                ep_secs = time.time() - ep_start
                summary = f"  ep={self.training.episode_count:04d}  time={ep_secs:.1f}s"
                print(summary)
                logger.info(summary)

                # Stop training when max episodes reached
                if self.training.episode_count >= Config.MAX_TRAINING_EPISODES:
                    training_complete = True
            
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"[ERROR] Exception during training: {e}")
            import traceback
            traceback.print_exc()
