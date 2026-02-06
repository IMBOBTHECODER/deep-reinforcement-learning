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
    wall_penalty_scale: float, stamina_penalty: float
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
    reward = goal_reward + distance_reward + proximity_reward + wall_penalty + stamina_penalty
    
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
        Observation for quadruped in AGENT-CENTERED WORLD.
        
        Agent COM is always at origin. Goal position is relative to agent.
        This simplifies learning since all observations are centered on the agent.
        
        Observation components (37D total):
        - Joint angles (12): positions of all 12 joints
        - Joint velocities (12): angular velocities
        - Foot contact (4): contact state of each foot [0, 1]
        - Orientation (3): pitch, yaw, roll (body tilt)
        - Center of mass (3): COM position (relative to body, usually ~0)
        - Goal relative (3): goal position - agent COM position (in agent-centered frame)
        
        Returns: (1, 37) tensor
        """
        from .entity import compute_center_of_mass
        
        # In agent-centered world, agent is at origin
        # Goal position is already relative (goal_pos_t - creature.pos = goal - agent_com)
        
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
        
        # RL models - QUADRUPED (37D observation, 12D action for motor torques)
        # WARNING: These are hardcoded here intentionally. Changing them requires editing simulate.py directly
        # because they have LARGE consequences:
        # - OBS_DIM=37 affects entity observations, rendering, evaluation visualization
        # - ACTION_DIM=12 affects motor control, physics simulation, evaluation
        NN_OBS_DIM = 37  # 12 angles + 12 velocities + 4 contacts + 3 orientation + 3 COM + 3 goal-rel
        NN_NUM_ACTIONS = 12  # Motor torques for 4 legs × 3 joints
        
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
        com_pos = compute_center_of_mass(creature.joint_angles, self.device, self.dtype)
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
                com_pos = compute_center_of_mass(creature.joint_angles, self.device, self.dtype)
                prev_com_dist = torch.norm(self.env.goal_pos_t[:2] - com_pos[:2])
            
            # Get next value estimate
            with torch.no_grad():
                next_obs = self.env.observe(creature)
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
        
        use_autocast = self.device.type == 'cuda'
        
        for _ in range(Config.PPO_EPOCHS):
            with torch.autocast(device_type='cuda', enabled=use_autocast):
                logits_list = []
                value_list = []
                
                h0, c0 = self.model.init_state(1, self.device, self.dtype)
                state = (h0.clone(), c0.clone())
                
                for t, obs in enumerate(obs_list):
                    (mu, log_std), value, state = self.model(obs, self.env.edge_indices[0], prev_state=state)
                    logits_list.append((mu, log_std))
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
    
    def train_on_trajectories_batch(self, trajectories):
        """Train on multiple trajectories from vectorized environments."""
        if not trajectories:
            return
        
        total_wm_loss = 0.0
        for trajectory in trajectories:
            wm_loss = self.train_world_model(trajectory)
            total_wm_loss += wm_loss
        avg_wm_loss = total_wm_loss / len(trajectories)
        logger.info(f"  Batch World Model Loss: {avg_wm_loss:.4f} ({len(trajectories)} envs)")
        
        self.model.train()
        
        all_rewards = []
        all_values = []
        all_next_values = []
        all_dones = []
        all_old_log_probs = []
        all_obs = []
        all_actions = []
        
        for trajectory in trajectories:
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
        
        use_autocast = self.device.type == 'cuda'
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(Config.PPO_EPOCHS):
            with torch.autocast(device_type='cuda', enabled=use_autocast):
                logits_list = []
                value_list = []
                
                step_idx = 0
                for traj_idx, obs_list in enumerate(all_obs):
                    h0, c0 = self.model.init_state(1, self.device, self.dtype)
                    state = (h0.clone(), c0.clone())
                    
                    for obs in obs_list:
                        (mu, log_std), value, state = self.model(obs, self.env.edge_indices[traj_idx], prev_state=state)
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
                a = torch.clamp(action_batch, -1.0 + self.physics.atanh_eps, 1.0 - self.physics.atanh_eps)
                u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
                
                log_prob_gaussian = -0.5 * ((u - mu) ** 2 / (std ** 2)).sum(dim=1)
                log_prob_gaussian = log_prob_gaussian - log_std_seq.sum(dim=1) - 0.5 * 12 * LOG_2PI  # 12D action space
                
                tanh_correction = -torch.log(1.0 - action_batch ** 2 + self.physics.log_eps).sum(dim=1)
                log_prob_seq = log_prob_gaussian + tanh_correction
                
                ratio = torch.exp(log_prob_seq - old_log_probs_batch)
                clipped_ratio = torch.clamp(ratio, 1.0 - Config.PPO_CLIP_RATIO, 1.0 + Config.PPO_CLIP_RATIO)
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
        
        batch_reward = sum(sum(float(t['reward']) for t in traj) for traj in trajectories) / len(trajectories)
        self.episode_rewards.append(batch_reward)
        msg = f"Batch Ep {self.episode_count}: avg_reward={batch_reward:.2f}, policy_loss={total_policy_loss/Config.PPO_EPOCHS:.4f}, value_loss={total_value_loss/Config.PPO_EPOCHS:.4f}"
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
            
            for i in idx:
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
                    # Vectorized: collect from all environments in parallel
                    for env_id in range(self.env.num_envs):
                        trajectory = self.training.collect_trajectory(max_steps=Config.MAX_STEPS_PER_EPISODE, env_id=env_id)
                        if trajectory:
                            trajectories_batch.append(trajectory)
                    
                    # Train once per batch
                    if trajectories_batch:
                        self.training.train_on_trajectories_batch(trajectories_batch)
                        self.training.episode_count += 1
                        trajectories_batch = []
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
