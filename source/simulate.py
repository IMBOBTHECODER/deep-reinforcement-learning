import numpy as np
import torch
import math
import logging
import os
import psutil
import GPUtil
from numba import jit
from .entity import EntityBelief, init_single_creature, WorldModel
from .physics import Quaternion, RigidBody, PhysicsEngine
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

# ===== JIT SIMULATION KERNEL (fused for maximum performance) =====
@jit(nopython=True)
def simulation_step(
    # Position & Velocity (agent)
    pos_x: float, pos_y: float, pos_z: float,
    vel_x: float, vel_y: float, vel_z: float,
    
    # Action (acceleration input)
    accel_x: float, accel_y: float, accel_z: float,
    
    # Goal position
    goal_x: float, goal_y: float, goal_z: float,
    
    # World bounds
    bound_min_x: float, bound_min_y: float, bound_min_z: float,
    bound_max_x: float, bound_max_y: float, bound_max_z: float,
    world_size_x: float, world_size_y: float, world_size_z: float,
    
    # Physics parameters
    accel_scale_xy: float, accel_scale_z: float, max_accel: float,
    max_vel: float, momentum_damping: float, gravity: float,
    terminal_vel_z: float, ground_level: float,
    ground_friction: float, air_friction: float, air_drag: float,
    
    # Reward parameters
    prev_dist: float, goal_threshold: float, proximity_threshold: float,
    distance_reward_scale: float, proximity_bonus_scale: float, goal_bonus: float,
    wall_penalty_scale: float
) -> tuple:
    """
    Complete simulation step kernel: movement, physics, reward, observation.
    All in one compiled function for zero boundary crossing overhead.
    
    Returns:
        (new_pos_x, new_pos_y, new_pos_z,
         new_vel_x, new_vel_y, new_vel_z,
         reward, curr_dist, wall_penalty,
         obs_rel_x, obs_rel_y, obs_rel_z,
         obs_abs_x, obs_abs_y, obs_abs_z)
    """
    
    # ===== PHYSICS UPDATE =====
    
    # 1. Scale acceleration by direction
    accel_x_scaled = accel_x * accel_scale_xy
    accel_y_scaled = accel_y * accel_scale_xy
    accel_z_scaled = accel_z * accel_scale_z
    
    # 2. Clamp acceleration magnitude
    accel_mag = (accel_x_scaled * accel_x_scaled + 
                 accel_y_scaled * accel_y_scaled + 
                 accel_z_scaled * accel_z_scaled) ** 0.5
    
    if accel_mag > max_accel:
        scale = max_accel / accel_mag
        accel_x_scaled *= scale
        accel_y_scaled *= scale
        accel_z_scaled *= scale
    
    # 3. Momentum damping
    new_vel_x = vel_x * (1.0 - momentum_damping)
    new_vel_y = vel_y * (1.0 - momentum_damping)
    new_vel_z = vel_z * (1.0 - momentum_damping)
    
    # 4. Apply acceleration
    new_vel_x += accel_x_scaled
    new_vel_y += accel_y_scaled
    new_vel_z += accel_z_scaled
    
    # 5. Gravity & ground collision
    on_ground = pos_z <= ground_level
    
    if on_ground:
        new_vel_z = max(0.0, new_vel_z)  # Don't penetrate ground
    else:
        new_vel_z -= gravity
        new_vel_z = max(new_vel_z, terminal_vel_z)  # Cap falling speed
    
    # 6. Apply drag
    if on_ground:
        horiz_speed = (new_vel_x * new_vel_x + new_vel_y * new_vel_y) ** 0.5
        if horiz_speed > 1e-6:
            drag_factor = max(0.0, 1.0 - (ground_friction * horiz_speed))
            new_vel_x *= drag_factor
            new_vel_y *= drag_factor
    else:
        vel_mag = (new_vel_x * new_vel_x + new_vel_y * new_vel_y + new_vel_z * new_vel_z) ** 0.5
        if vel_mag > 1e-6:
            # Quadratic air drag
            drag_mag = air_drag * vel_mag * vel_mag
            drag_scale = -drag_mag / vel_mag
            new_vel_x += new_vel_x * drag_scale
            new_vel_y += new_vel_y * drag_scale
            new_vel_z += new_vel_z * drag_scale
            
            # Air friction
            new_vel_x *= (1.0 - air_friction)
            new_vel_y *= (1.0 - air_friction)
    
    # 7. Clamp velocity
    vel_mag = (new_vel_x * new_vel_x + new_vel_y * new_vel_y + new_vel_z * new_vel_z) ** 0.5
    if vel_mag > max_vel:
        scale = max_vel / vel_mag
        new_vel_x *= scale
        new_vel_y *= scale
        new_vel_z *= scale
    
    # 8. Update position
    new_pos_x = pos_x + new_vel_x
    new_pos_y = pos_y + new_vel_y
    new_pos_z = pos_z + new_vel_z
    
    # 9. Clamp to boundaries (inline to avoid function call)
    clamped_x = max(bound_min_x, min(new_pos_x, bound_max_x))
    clamped_y = max(bound_min_y, min(new_pos_y, bound_max_y))
    clamped_z = max(bound_min_z, min(new_pos_z, bound_max_z))
    
    penetration = abs(new_pos_x - clamped_x) + abs(new_pos_y - clamped_y) + abs(new_pos_z - clamped_z)
    wall_penalty = -wall_penalty_scale * penetration
    
    # ===== OBSERVATION COMPUTATION =====
    # Inline normalization (no function call)
    inv_x = 1.0 / world_size_x
    inv_y = 1.0 / world_size_y
    inv_z = 1.0 / world_size_z
    
    obs_rel_x = (goal_x - clamped_x) * inv_x
    obs_rel_y = (goal_y - clamped_y) * inv_y
    obs_rel_z = (goal_z - clamped_z) * inv_z
    
    obs_abs_x = clamped_x * inv_x
    obs_abs_y = clamped_y * inv_y
    obs_abs_z = clamped_z * inv_z
    
    # ===== REWARD COMPUTATION =====
    # Distance to goal
    dx = clamped_x - goal_x
    dy = clamped_y - goal_y
    dz = clamped_z - goal_z
    curr_dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    
    # Distance change reward
    distance_delta = prev_dist - curr_dist
    if distance_delta > 0:
        distance_reward = distance_delta * distance_reward_scale
    else:
        distance_reward = distance_delta * distance_reward_scale * 2.0  # 2x penalty
    
    # Goal bonus
    goal_reached = 1.0 if curr_dist < goal_threshold else 0.0
    goal_reward = goal_reached * goal_bonus
    
    # Proximity bonus
    proximity_reached = 1.0 if curr_dist < proximity_threshold else 0.0
    proximity_reward = proximity_reached * (proximity_threshold - curr_dist) * proximity_bonus_scale
    
    # Total reward
    reward = goal_reward + distance_reward + proximity_reward + wall_penalty
    
    return (
        clamped_x, clamped_y, clamped_z,
        new_vel_x, new_vel_y, new_vel_z,
        reward, curr_dist, wall_penalty,
        obs_rel_x, obs_rel_y, obs_rel_z,
        obs_abs_x, obs_abs_y, obs_abs_z
    )


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
        
        # Goal position (single source of truth on device)
        self.goal_pos_t = torch.tensor([float(self.w - 5), float(self.h - 5), float(self.d - 1)], device=device, dtype=dtype)
        
        # Multi-environment support
        self.use_vectorized = Config.USE_VECTORIZED_ENV
        self.num_envs = Config.NUM_ENVS if Config.NUM_ENVS is not None else detect_available_envs() if Config.USE_VECTORIZED_ENV else 1
        
        if self.use_vectorized:
            logger.info(f"Vectorized environment enabled with {self.num_envs} parallel environments")
        else:
            logger.info("Single environment training")
        
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
        """Calculate Euclidean distance from creature to goal in 3D space."""
        dx = creature.pos[0] - self.goal_pos_t[0]
        dy = creature.pos[1] - self.goal_pos_t[1]
        dz = creature.pos[2] - self.goal_pos_t[2]
        dist_sq = dx**2 + dy**2 + dz**2
        return torch.sqrt(dist_sq)
    
    def spawn_random_goal(self, env_id=0):
        """Spawn a new goal at a random location on the map (in-place update)."""
        x = np.random.uniform(20, self.w - 20)
        y = np.random.uniform(20, self.h - 20)
        z = np.random.uniform(0, self.d - 5)
        self.goal_pos_t[0] = float(x)
        self.goal_pos_t[1] = float(y)
        self.goal_pos_t[2] = float(z)
        if env_id < len(self.creatures):
            self.prev_distances[env_id] = self._distance_to_goal(self.creatures[env_id])
    
    def observe(self, creature):
        """
        Observation for quadruped in AGENT-CENTERED WORLD (37D total).
        See docs/QUADRUPED_BALANCE_TASK.md for component breakdown.
        
        Returns: (1, 37) tensor
        """
        from .entity import compute_center_of_mass
        
        obs_list = [
            creature.joint_angles,                                    # (12,)
            creature.joint_velocities,                                # (12,)
            creature.foot_contact,                                    # (4,)
            creature.orientation,                                     # (3,)
            torch.zeros(3, device=self.device, dtype=self.dtype),   # (3,) COM at origin in local frame
            self.goal_pos_t - creature.pos                           # (3,) goal relative to agent
        ]
        
        obs = torch.cat(obs_list).unsqueeze(0).to(self.device)  # (1, 37)
        return obs


class TrainingEngine:
    """Manages RL training: data collection, PPO, world model."""
    
    def __init__(self, device, dtype, environment, physics_engine):
        self.device = device
        self.dtype = dtype
        self.env = environment
        self.physics = physics_engine
        
        # RL models - QUADRUPED (37D observation, 12D motor torques)
        # WARNING: OBS_DIM=37 and ACTION_DIM=12 are hardcoded; changes affect multiple modules
        NN_OBS_DIM = 37
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
        
        # World model
        self.world_model = WorldModel(
            obs_dim=NN_OBS_DIM,
            action_dim=NN_NUM_ACTIONS,
            latent_dim=Config.WORLD_MODEL_LATENT_DIM,
            hidden_dim=Config.WORLD_MODEL_HIDDEN_DIM
        ).to(device)
        
        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=Config.WORLD_MODEL_LR,
            weight_decay=Config.WORLD_MODEL_WEIGHT_DECAY
        )
        
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
        batch_size = num_envs
        
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
        
        # Main loop: run all environments until they all reach max_steps
        for global_step in range(max_steps):
            # Collect observations from all environments
            obs_list = []
            for env_id in range(num_envs):
                if step_counts[env_id] < max_steps:
                    creature = self.env.creatures[env_id]
                    obs = self.env.observe(creature)  # (1, 37)
                    obs_list.append(obs)
            
            if not obs_list:
                break  # All environments done
            
            # Stack into batch: (batch_size, 37)
            obs_batch = torch.cat(obs_list, dim=0)  # (num_active_envs, 37)
            
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
                    
                    (mu, log_std), values, (new_h, new_c) = self.model(
                        obs_batch, edge_idx, prev_state=(h_batch, c_batch)
                    )
                    
                    # Sample actions
                    std = torch.exp(log_std)
                    u = mu + torch.randn_like(mu) * std
                    actions = torch.tanh(u)  # (batch, 12)
                    
                    # Compute log probs
                    log_prob_gaussian = -0.5 * ((u - mu) ** 2 / (std ** 2)).sum(dim=1)
                    log_prob_gaussian = log_prob_gaussian - log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI
                    tanh_correction = -torch.log(1.0 - actions ** 2 + self.log_eps).sum(dim=1)
                    log_probs = log_prob_gaussian + tanh_correction
                    
                    # Process results for each environment
                    env_id_active = 0
                    for env_id in range(num_envs):
                        if step_counts[env_id] < max_steps:
                            creature = self.env.creatures[env_id]
                            
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
                            
                            # Scale action to motor torques
                            motor_torques = action_i[0] * 5.0
                            
                            # Compute reward
                            reward, com_dist, _ = self.physics._compute_reward(
                                creature, motor_torques, self.env.goal_pos_t
                            )
                            
                            # Check if done
                            done = float(com_dist) < self.physics.com_distance_threshold
                            if done:
                                self.env.spawn_random_goal(env_id)
                                # NOTE: creature.rnn_state intentionally NOT reset here
                                # This is a CONTINUING task - the agent remembers across goals
                                # Training will handle episode boundaries separately
                            
                            # Get next value
                            with torch.no_grad():
                                next_obs = self.env.observe(creature)
                                # RNN state carries forward to next goal for continuity
                                (_, _), next_value, _ = self.model(
                                    next_obs, edge_idx, prev_state=creature.rnn_state
                                )
                            
                            # Store transition
                            trajectories[env_id].append({
                                'obs': obs_list[env_id_active],
                                'action': action_i,
                                'reward': reward,
                                'value': value_i.squeeze().detach(),
                                'next_value': next_value.squeeze().detach(),
                                'done': float(done),
                                'old_log_prob': log_prob_i.detach(),
                            })
                            
                            step_counts[env_id] += 1
                            env_id_active += 1
                
                except RuntimeError as e:
                    # Fallback: if batch processing fails, fall back to sequential
                    logger.warning(f"Batch processing failed: {e}, falling back to sequential")
                    return self.collect_trajectory_sequential(max_steps)
        
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
        
        # Reset RNN state
        h0, c0 = self.model.init_state(1, self.device, self.dtype)
        creature.rnn_state = (h0, c0)
        
        # Spawn random goal within reachable distance
        self.env.spawn_random_goal(env_id)
        
        # Store initial COM distance for tracking
        from .entity import compute_center_of_mass
        com_pos = compute_center_of_mass(creature.joint_angles, creature.orientation)
        prev_com_dist = torch.norm(self.env.goal_pos_t[:2] - com_pos[:2])
        
        while step_count < max_steps:
            # Get observation
            obs = self.env.observe(creature)
            
            # Get action from policy (12D motor torques)
            with torch.no_grad():
                (mu, log_std), value, new_state = self.model(obs, edge_idx, prev_state=creature.rnn_state)
            
            creature.rnn_state = new_state
            
            # Sample action from policy (tanh-squashed Gaussian)
            std = torch.exp(log_std)
            u = mu + torch.randn_like(mu) * std
            action = torch.tanh(u)  # (1, 12) squashed to [-1, 1]
            
            # Compute log probability
            log_prob_gaussian = -0.5 * ((u - mu) ** 2 / (std ** 2)).sum(dim=1)
            log_prob_gaussian = log_prob_gaussian - log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI
            tanh_correction = -torch.log(1.0 - action ** 2 + self.log_eps).sum(dim=1)
            log_prob = log_prob_gaussian + tanh_correction
            
            # Scale action to motor torques ([-5, 5] N*m range)
            motor_torques = action[0] * 5.0  # (12,) scaled torques
            
            # Apply physics and get reward
            reward, com_dist, stability_metrics = self.physics._compute_reward(
                creature, motor_torques, self.env.goal_pos_t
            )
            
            # Check if goal reached (COM within threshold)
            done = float(com_dist) < self.physics.com_distance_threshold
            if done:
                self.env.spawn_random_goal(env_id)
                com_pos = compute_center_of_mass(creature.joint_angles, creature.orientation)
                prev_com_dist = torch.norm(self.env.goal_pos_t[:2] - com_pos[:2])
                # NOTE: creature.rnn_state intentionally NOT reset here
                # This is a CONTINUING task - the agent remembers across goals
                # Training will handle episode boundaries separately
            
            # Get next value estimate
            with torch.no_grad():
                next_obs = self.env.observe(creature)
                # RNN state carries forward to next goal for continuity
                (next_mu, next_log_std), next_value, _ = self.model(
                    next_obs, edge_idx, prev_state=creature.rnn_state
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
            })
            
            step_count += 1
        
        self.step_count = step_count
        return trajectory
    
    def train_on_trajectory(self, trajectory):
        """Train on a collected trajectory using PPO for quadruped motor control."""
        if not trajectory:
            return
        
        wm_loss = self.train_world_model(trajectory)
        logger.info(f"  World Model Loss: {wm_loss:.4f}")
        
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
                logits_list = []
                value_list = []
                
                h0, c0 = self.model.init_state(1, self.device, self.dtype)
                state = (h0.clone(), c0.clone())
                
                for t, obs in enumerate(obs_list):
                    (mu, log_std), value, state = self.model(obs, self.env.edge_indices[0], prev_state=state)
                    logits_list.append((mu, log_std))
                    # During training: treat goal_reached (done) as episode boundary
                    # Reset RNN state when done=1 (episodic training)
                    value_list.append(value)
                    
                    h, c = state
                    mask = (1.0 - dones[t]).view(1, 1)
                    state = (h * mask, c * mask)
                
                mu_seq = torch.cat([m for m, _ in logits_list], dim=0)  # (T, 12)
                log_std_seq = torch.cat([s for _, s in logits_list], dim=0)  # (T, 12)
                value_seq = torch.cat(value_list, dim=0)
                
                action_batch = torch.cat(action_list, dim=0)  # (T, 12)
                mu = mu_seq
                std = torch.exp(log_std_seq)
                
                # Stable atanh for 12D action space
                a = torch.clamp(action_batch, -1.0 + self.physics.atanh_eps, 1.0 - self.physics.atanh_eps)
                u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
                
                # Log probability for 12D Gaussian policy
                log_prob_gaussian = -0.5 * ((u - mu) ** 2 / (std ** 2)).sum(dim=1)
                log_prob_gaussian = log_prob_gaussian - log_std_seq.sum(dim=1) - 0.5 * 12 * LOG_2PI  # 12D
                
                # Tanh correction for 12D
                tanh_correction = -torch.log(1.0 - action_batch ** 2 + self.physics.log_eps).sum(dim=1)
                log_prob_seq = log_prob_gaussian + tanh_correction
                
                ratio = torch.exp(log_prob_seq - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - Config.PPO_CLIP_RATIO, 1.0 + Config.PPO_CLIP_RATIO)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
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
        
        episode_reward = sum(float(t['reward']) for t in trajectory)
        self.episode_rewards.append(episode_reward)
        msg = f"Ep {self.episode_count}: reward={episode_reward:.2f}, policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
        logger.info(msg)
        print(msg)
    
    def train_on_trajectories_batch(self, trajectories_wrapped):
        """Train on multiple trajectories from vectorized environments.
        
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
        
        total_wm_loss = 0.0
        for traj_data in trajectories_wrapped:
            wm_loss = self.train_world_model(traj_data['trajectory'])
            total_wm_loss += wm_loss
        avg_wm_loss = total_wm_loss / len(trajectories_wrapped)
        logger.info(f"  Batch World Model Loss: {avg_wm_loss:.4f} ({len(trajectories_wrapped)} envs)")
        
        self.model.train()
        
        all_rewards = []
        all_values = []
        all_next_values = []
        all_dones = []
        all_old_log_probs = []
        all_obs = []
        all_actions = []
        all_edge_indices = []  # NEW: Store edge indices aligned with trajectories
        
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
            
            all_rewards.append(rewards)
            all_values.append(values)
            all_next_values.append(next_values)
            all_dones.append(dones)
            all_old_log_probs.append(old_log_probs)
            all_obs.append(obs_list)
            all_actions.append(action_list)
            all_edge_indices.append(edge_idx)  # NEW: Store edge index for this trajectory
        
        rewards_batch = torch.cat(all_rewards)
        values_batch = torch.cat(all_values)
        next_values_batch = torch.cat(all_next_values)
        dones_batch = torch.cat(all_dones)
        old_log_probs_batch = torch.cat(all_old_log_probs)
        
        deltas = rewards_batch + self.gamma * next_values_batch.squeeze() * (1 - dones_batch) - values_batch.squeeze()
        advantages_raw = torch.zeros_like(rewards_batch)
        gae = 0
        for t in reversed(range(len(rewards_batch))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones_batch[t]) * gae
            advantages_raw[t] = gae
        
        returns_batch = advantages_raw + values_batch.squeeze()
        advantages_batch = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        
        # Debug: log shapes and value ranges
        logger.info(f"[SHAPES] advantages_batch: {advantages_batch.shape}, min={advantages_batch.min():.4f}, max={advantages_batch.max():.4f}")
        logger.info(f"[SHAPES] returns_batch: {returns_batch.shape}, min={returns_batch.min():.4f}, max={returns_batch.max():.4f}")
        logger.info(f"[SHAPES] old_log_probs_batch: {old_log_probs_batch.shape}, min={old_log_probs_batch.min():.4f}, max={old_log_probs_batch.max():.4f}")
        logger.info(f"[SHAPES] values_batch: {values_batch.shape}, min={values_batch.min():.4f}, max={values_batch.max():.4f}")
        logger.info(f"[SHAPES] next_values_batch: {next_values_batch.shape}, min={next_values_batch.min():.4f}, max={next_values_batch.max():.4f}")
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for epoch in range(Config.PPO_EPOCHS):
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                logits_list = []
                value_list = []
                
                step_idx = 0
                for traj_idx, obs_list in enumerate(all_obs):
                    h0, c0 = self.model.init_state(1, self.device, self.dtype)
                    state = (h0.clone(), c0.clone())
                    
                    # Use edge_index stored with this trajectory, not global index
                    edge_idx = all_edge_indices[traj_idx]
                    
                    for obs in obs_list:
                        (mu, log_std), value, state = self.model(obs, edge_idx, prev_state=state)
                        logits_list.append((mu, log_std))
                        value_list.append(value)
                        
                        h, c = state
                        mask = (1.0 - dones_batch[step_idx]).view(1, 1)
                        state = (h * mask, c * mask)
                        step_idx += 1
                
                mu_seq = torch.cat([m for m, _ in logits_list], dim=0)
                log_std_seq = torch.cat([s for _, s in logits_list], dim=0)
                value_seq = torch.cat(value_list, dim=0)
                
                all_actions_flat = []
                for action_list in all_actions:
                    all_actions_flat.extend(action_list)
                action_batch = torch.cat(all_actions_flat, dim=0)
                
                mu = mu_seq
                std = torch.exp(log_std_seq)
                
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] mu: min={mu.min():.6f}, max={mu.max():.6f}, has_nan={torch.isnan(mu).any()}")
                    logger.info(f"[EPOCH {epoch}] std: min={std.min():.6f}, max={std.max():.6f}, has_nan={torch.isnan(std).any()}")
                
                a = torch.clamp(action_batch, -1.0 + self.physics.atanh_eps, 1.0 - self.physics.atanh_eps)
                
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] action_batch (clamped): min={a.min():.6f}, max={a.max():.6f}, has_nan={torch.isnan(a).any()}")
                
                u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
                
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] u (inverse_tanh): min={u.min():.6f}, max={u.max():.6f}, has_nan={torch.isnan(u).any()}")
                
                # First compute the Gaussian log prob
                action_diff = u - mu  # (T, 12)
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] (u - mu): min={action_diff.min():.6f}, max={action_diff.max():.6f}, has_nan={torch.isnan(action_diff).any()}")
                
                gaussian_term = -0.5 * ((action_diff) ** 2 / (std ** 2)).sum(dim=1)  # (T,)
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] gaussian_term: min={gaussian_term.min():.6f}, max={gaussian_term.max():.6f}, has_nan={torch.isnan(gaussian_term).any()}")
                
                log_prob_gaussian = gaussian_term - log_std_seq.sum(dim=1) - 0.5 * 12 * LOG_2PI
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] log_prob_gaussian: min={log_prob_gaussian.min():.6f}, max={log_prob_gaussian.max():.6f}, has_nan={torch.isnan(log_prob_gaussian).any()}")
                
                # Now tanh correction  
                action_sq_term = 1.0 - action_batch ** 2
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] (1 - action^2): min={action_sq_term.min():.6f}, max={action_sq_term.max():.6f}, has_nan={torch.isnan(action_sq_term).any()}")
                
                tanh_correction = -torch.log(action_sq_term + self.physics.log_eps).sum(dim=1)
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] tanh_correction: min={tanh_correction.min():.6f}, max={tanh_correction.max():.6f}, has_nan={torch.isnan(tanh_correction).any()}")
                
                log_prob_seq = log_prob_gaussian + tanh_correction
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] log_prob_seq (final): min={log_prob_seq.min():.6f}, max={log_prob_seq.max():.6f}, has_nan={torch.isnan(log_prob_seq).any()}")
                    logger.info(f"[EPOCH {epoch}] old_log_probs_batch: min={old_log_probs_batch.min():.6f}, max={old_log_probs_batch.max():.6f}, has_nan={torch.isnan(old_log_probs_batch).any()}")
                
                log_prob_diff = log_prob_seq - old_log_probs_batch
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] log_prob_diff: min={log_prob_diff.min():.6f}, max={log_prob_diff.max():.6f}, has_nan={torch.isnan(log_prob_diff).any()}")
                
                ratio = torch.exp(log_prob_diff)
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] ratio: min={ratio.min():.6f}, max={ratio.max():.6f}, has_nan={torch.isnan(ratio).any()}, has_inf={torch.isinf(ratio).any()}")
                
                clipped_ratio = torch.clamp(ratio, 1.0 - Config.PPO_CLIP_RATIO, 1.0 + Config.PPO_CLIP_RATIO)
                
                if epoch == 0:
                    logger.info(f"[EPOCH {epoch}] advantages_batch: min={advantages_batch.min():.6f}, max={advantages_batch.max():.6f}, has_nan={torch.isnan(advantages_batch).any()}")
                    logger.info(f"[EPOCH {epoch}] ratio * advantages: min={(ratio * advantages_batch).min():.6f}, max={(ratio * advantages_batch).max():.6f}, has_nan={torch.isnan(ratio * advantages_batch).any()}")
                    logger.info(f"[EPOCH {epoch}] clipped_ratio * advantages: min={(clipped_ratio * advantages_batch).min():.6f}, max={(clipped_ratio * advantages_batch).max():.6f}, has_nan={torch.isnan(clipped_ratio * advantages_batch).any()}")
                
                policy_loss = -torch.min(ratio * advantages_batch, clipped_ratio * advantages_batch).mean()
                
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
        
        # Calculate average reward across all trajectories
        batch_reward = sum(sum(float(t['reward']) for t in traj_data['trajectory']) for traj_data in trajectories_wrapped) / len(trajectories_wrapped)
        self.episode_rewards.append(batch_reward)
        
        # Log debugging info
        msg = f"Batch Ep {self.episode_count}: avg_reward={batch_reward:.2f}"
        if not torch.isnan(policy_loss) and not torch.isnan(value_loss):
            msg += f", policy_loss={total_policy_loss/Config.PPO_EPOCHS:.4f}, value_loss={total_value_loss/Config.PPO_EPOCHS:.4f}"
        logger.info(msg)
        print(msg)
    
    def train_world_model(self, trajectory):
        """Train world model on trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        self.world_model.train()
        
        total_loss = 0.0
        num_batches = max(1, len(trajectory) // 32)
        
        for step in range(num_batches):
            idx = torch.randint(0, len(trajectory) - 1, (32,))
            
            batch_obs = []
            batch_actions = []
            batch_next_obs = []
            batch_rewards = []
            
            # Convert tensor indices to Python ints for list indexing
            for i in idx.tolist():
                batch_obs.append(trajectory[i]['obs'])
                batch_actions.append(trajectory[i]['action'])
                batch_next_obs.append(trajectory[i + 1]['obs'])
                batch_rewards.append(trajectory[i]['reward'].unsqueeze(0))
            
            obs_batch = torch.cat(batch_obs, dim=0)
            actions_batch = torch.cat(batch_actions, dim=0)
            rewards_batch = torch.cat(batch_rewards, dim=0).squeeze(-1)  # Shape: [32]
            
            next_latent, pred_rewards, pred_dones, recon_obs = self.world_model(obs_batch, actions_batch)
            
            recon_loss = torch.nn.functional.mse_loss(recon_obs, obs_batch)
            reward_loss = torch.nn.functional.mse_loss(pred_rewards.squeeze(-1), rewards_batch)
            wm_loss = recon_loss + reward_loss
            
            self.world_model_optimizer.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_model_optimizer.step()
            
            total_loss += wm_loss.detach().item()
        
        return total_loss / num_batches


class System:
    """Lightweight orchestrator that coordinates all components."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
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
