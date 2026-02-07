"""Configuration settings for RL training system - Quadruped Balance Task."""
from pathlib import Path
import numpy as np


class Config:
    """Training and system configuration."""
    
    # Core training parameters
    DIM = (64, 32, 64)  # World dimensions (larger for balance task)
    MAX_STEPS_PER_EPISODE = 1000
    MAX_TRAINING_EPISODES = 50
    LOAD_CHECKPOINT = False
    CHECKPOINT_DIR = Path("checkpoint")
    
    # ============ VECTORIZED MULTI-ENVIRONMENT TRAINING ============
    USE_VECTORIZED_ENV = True
    NUM_ENVS = None  # Auto-detect based on available memory
    MAX_ENVS = 64
    MIN_ENVS = 2
    MAX_DATA_THRESHOLD_MB = 8192
    
    # Memory model updated for quadruped observation (37D)
    # Observation: (1, 37) float32 = 148 bytes per observation
    OBS_BYTES = 148  # (1, 37) float32 quadruped observation
    PER_STEP_BYTES = 300  # obs + action(12) + logprob + value + reward + done + buffers
    ROLLOUT_STEPS = 256
    MEMORY_OVERHEAD_MB = 256
    
    # Evaluation & rendering
    EVAL_EPISODES = 2
    RUN_EVALUATION = True

    SCALE = 32
    TARGET_FPS = 60
    EVAL_STEPS_PER_SEC = 20
    
    # Agent configuration - QUADRUPED (4 legs, 3 joints each)
    ENTITY_TYPES = {
        1: {"type": "agent", "color": np.array((255, 0, 0, 255), dtype=np.uint8), "size": (1, 1, 1), "speed": 1.0},
        2: {"type": "goal", "color": np.array((0, 255, 0, 255), dtype=np.uint8), "size": (2, 2, 2), "speed": 0.0},
    }
    
    # ============ NEURAL NETWORK ARCHITECTURE ============
    EMBED_DIM = 128
    GAT_OUT_DIM = 64
    GAT_HEADS = 4
    LSTM_HIDDEN = 256
    
    # ============ PPO HYPERPARAMETERS ============
    PPO_EPOCHS = 4
    PPO_CLIP_RATIO = 0.2
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    LR = 3e-4
    
    # ============ TRAINING STABILITY ============
    GRAD_CLIP_NORM = 0.5  # Prevent exploding gradients
    VALUE_CLIP_RANGE = 0.2  # Clip value function updates
    
    # ============ MULTI-CORE PROCESSING ============
    # Physics processing distributed across CPU cores
    NUM_PHYSICS_THREADS = None  # None = auto-detect (typically num_cpus - 1)
    
    # ============ BALANCE TASK REWARD SHAPING ============
    # Goal: Keep center of mass (COM) close to goal position
    # Reward = exp(-distance) + tilt_penalty + contact_reward - energy_cost
    CONTACT_REWARD = 0.1  # Reward per foot in contact with ground
    ENERGY_PENALTY = 0.01  # Penalty for high motor torques
    TILT_PENALTY = 0.5  # Penalty for tilting beyond MAX_PITCH_ROLL
    COM_DISTANCE_THRESHOLD = 0.3  # Reference distance (for documentation)
    
    # Legacy params (kept for compatibility, not used in balance task)
    GOAL_DISTANCE_THRESHOLD = 0.3
    PROXIMITY_THRESHOLD = 10.0
    PROXIMITY_BONUS_SCALE = 0.1
    GOAL_BONUS = 10.0
    DISTANCE_REWARD_SCALE = 0.5
    
    
    # ============ PHYSICS SYSTEM - QUADRUPED ============
    # Physics parameters (tuned for stability)
    DT = 0.01  # Timestep: 100 Hz physics simulation
    JOINT_DAMPING = 0.1  # Increased damping for smoother motion
    MAX_TORQUE = 5.0  # Maximum motor torque (N*m)
    SEGMENT_LENGTH = 0.1  # Length of each leg segment (3 segments = 0.3m leg)
    MAX_JOINT_VELOCITY = 10.0  # rad/s
    
    # Ground properties
    GRAVITY = 9.81  # Earth gravity (m/s²)
    GROUND_LEVEL = 0.0  # Z-coordinate of ground plane
    GROUND_FRICTION_COEFFICIENT = 0.9  # Rubber/bio pads on concrete
    FOOT_HEIGHT_THRESHOLD = 0.05  # Distance below ground to register contact
    
    # Contact physics (spring-damper model)
    CONTACT_STIFFNESS = 500.0  # Stiff ground (concrete-like, N/m per foot)
    CONTACT_DAMPING = 0.15  # Ground damping (realistic for impact)
    CONTACT_RESTITUTION = 0.1  # Very low bounce (Earth materials)
    
    # Rigid body properties
    BODY_MASS = 5.0  # kg, quadruped torso mass
    BODY_DIMENSIONS = (0.5, 0.2, 0.3)  # (length, width, height) for inertia tensor
    BODY_INITIAL_HEIGHT = 0.3  # Initial Z position
    
    # Numerical tolerances
    ATANH_EPSILON = 1e-6  # For stable tanh in action squashing
    LOG_EPSILON = 1e-6  # For stable log operations
    MAX_ANGULAR_VELOCITY = 50.0  # rad/s, clamp spinning
    
    # Balance control
    MAX_PITCH_ROLL = 0.5  # rad, max allowed tilt before penalty
    
    # Advanced physics (legacy, not used in new system)
    MOMENTUM_DAMPING = 0.02
    TERMINAL_VELOCITY_Z = -30.0
    MAX_ACCELERATION = 2.0
    ACTION_SCALE_Z = 0.2
    MAX_VELOCITY = 10.0
    AIR_FRICTION_COEFFICIENT = 0.03
    AIR_DRAG_COEFFICIENT = 0.001
    
    # ============ DREAMER V3 WORLD MODEL ============
    WORLD_MODEL_LATENT_DIM = 128
    WORLD_MODEL_HIDDEN_DIM = 256
    WORLD_MODEL_LR = 1e-3
    WORLD_MODEL_WEIGHT_DECAY = 1e-6
    IMAGINATION_HORIZON = 15
    WORLD_MODEL_LOSS_SCALE = 1.0

    @classmethod
    def auto_num_envs(cls, available_memory_mb=None):
        """
        Calculate optimal number of environments based on realistic memory model.
        
        Args:
            available_memory_mb: Total available memory (GPU + CPU) in MB.
                                If None, auto-detects via psutil/GPUtil.
        
        Returns:
            int: Number of environments to use (between MIN_ENVS and MAX_ENVS)
        """
        import psutil
        import torch
        
        if available_memory_mb is None:
            # Auto-detect available memory
            cpu_mem_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            gpu_mem_mb = 0
            if torch.cuda.is_available():
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    gpu_mem_mb = sum(gpu.memoryFree for gpu in gpus) if gpus else 0
                except:
                    pass
            
            available_memory_mb = cpu_mem_mb + gpu_mem_mb
        
        # Budget: use at most MAX_DATA_THRESHOLD_MB, leaving overhead for OS
        budget_mb = min(cls.MAX_DATA_THRESHOLD_MB, available_memory_mb) - cls.MEMORY_OVERHEAD_MB
        budget_mb = max(budget_mb, 128)  # Ensure at least 128 MB for training
        
        # Per-environment memory cost: rollout buffer size
        per_env_mb = (cls.ROLLOUT_STEPS * cls.PER_STEP_BYTES) / (1024 * 1024)
        
        # Calculate how many environments fit
        num_envs = int(budget_mb / max(per_env_mb, 1e-6))
        num_envs = max(cls.MIN_ENVS, min(cls.MAX_ENVS, num_envs))
        
        return num_envs