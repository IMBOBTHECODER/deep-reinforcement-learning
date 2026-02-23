"""Configuration settings for RL training system - Quadruped Balance Task."""
from pathlib import Path
import numpy as np


class Config:
    """Training and system configuration."""
    
    # Core training parameters
    DIM = (64, 32, 64)  # World dimensions (larger for balance task)
    MAX_STEPS_PER_EPISODE = 1024
    MAX_TRAINING_EPISODES = 128
    LOAD_CHECKPOINT = False
    CHECKPOINT_DIR = Path("checkpoint")
    
    # ============ VECTORIZED MULTI-ENVIRONMENT TRAINING ============
    # Always enabled: Multiple agents train in parallel for efficiency
    NUM_ENVS = None  # Auto-detect based on available memory
    MAX_ENVS = 16
    MIN_ENVS = 2
    MAX_DATA_THRESHOLD_MB = 8192
    
    # Memory model updated for quadruped observation (34D)
    # Observation: (1, 34) float32 = 136 bytes per observation
    OBS_BYTES = 136  # (1, 34) float32 quadruped observation
    PER_STEP_BYTES = 280  # obs + action(12) + logprob + value + reward + done + buffers
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
    # Always enabled: CPU threads handle physics when GPU not available
    NUM_PHYSICS_THREADS = None  # None = auto-detect (typically num_cpus - 1)
    PHYSICS_BATCH_SIZE = 4  # Physics steps batched together per thread
    
    # ============ GPU ACCELERATION & VECTORIZATION (Numba CUDA) ============
    # Phase 4b: GPU-accelerated vectorized physics for 1000+ environments
    GPU_THREADS_PER_BLOCK = 1024  # Threads per block (1024 for RTX cards, 512 for older)
    GPU_MAX_BLOCKS = 32  # Maximum thread blocks for occupancy
    VECTORIZED_PHYSICS = True  # Enable batched physics kernel (1000+ envs on GPU)
    VECTORIZED_BATCH_SIZE = 1024  # Batch size for GPU physics (environments per kernel call)
    
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
    DT = 0.004  # Timestep: 250 Hz simulation (upgraded from 100 Hz for finer contact resolution)
    JOINT_DAMPING = 0.1  # Increased damping for smoother motion
    MAX_TORQUE = 5.0  # Maximum motor torque (N*m)
    SEGMENT_LENGTH = 0.1  # Length of each leg segment (3 segments = 0.3m leg)
    MAX_JOINT_VELOCITY = 10.0  # rad/s
    
    # Ground properties
    GRAVITY = 9.81  # Earth gravity (m/s²)
    GROUND_LEVEL = 0.0  # Z-coordinate of ground plane
    GROUND_FRICTION_COEFFICIENT = 0.9  # Rubber/bio pads on concrete
    FOOT_HEIGHT_THRESHOLD = 0.05  # Distance below ground to register contact
    
    # Contact physics (spring-damper model with restitution & friction cones)
    CONTACT_STIFFNESS = 500.0  # Stiff ground (concrete-like, N/m per foot)
    CONTACT_DAMPING = 0.15  # Ground damping (realistic for impact)
    CONTACT_RESTITUTION = 0.1  # Coefficient of restitution (0=no bounce, 1=perfect). 0.1 = slight bounce
    
    # ============ FRICTION CONES (Phase 3: Directional Friction Constraint) ============
    # Friction cone prevents unrealistic sliding: ||F_tangent|| <= mu * F_normal
    USE_FRICTION_CONES = True  # Enable 3D friction cones (prevents sideways sliding)
    FRICTION_CONE_DAMPING = 0.3  # Additional damping within friction cone for numerical stability
    
    # ============ ACTUATOR DYNAMICS (Phase 1 Realism Improvement) ============
    # First-order response lag: τ_actual = τ_actual + (τ_commanded - τ_actual) * (dt / τ_response)
    ACTUATOR_RESPONSE_TIME = 0.01  # seconds (10ms lag, typical for servo motors)
    # If 0.0, disabled (direct torque application, legacy behavior)
    
    # ============ FRICTION MODEL (Phase 2 Realism Improvement) ============
    FRICTION_MODEL = "coulomb+viscous"  # Options: "simple" (legacy), "coulomb", "coulomb+viscous"
    FRICTION_COEFFICIENT_STATIC = 0.9   # μ_s: resistance to initial slip
    FRICTION_COEFFICIENT_KINETIC = 0.85  # μ_k: resistance during active slip (slightly lower)
    FRICTION_VISCOUS_DAMPING = 0.05  # η: viscous damping during slip (N·s/m)
    FRICTION_SLIP_VELOCITY_THRESHOLD = 0.01  # m/s: transition between static/kinetic
    
    # ============ ENERGY ACCOUNTING (Phase 4 Strategic Improvement) ============
    TRACK_ENERGY_CONSUMPTION = True  # If True, accumulate electrical energy spent
    MOTOR_EFFICIENCY = 0.80  # Mechanical efficiency: 0-100%, typical servos 60-90%
    
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
    ACTION_SCALE_Z = 0.2  # Scale factor for Z-axis actions (reduce vertical thrashing)
    
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