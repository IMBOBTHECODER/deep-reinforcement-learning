import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from typing import Tuple, Optional
import math


# Module-level constants for kinematic offsets (avoid rebuilding on every call)
_HIP_OFFSETS_RAW = {
    0: [0.15, 0.0, -0.1],    # Front-left
    1: [0.15, 0.0, 0.1],     # Front-right
    2: [-0.15, 0.0, -0.1],   # Back-left
    3: [-0.15, 0.0, 0.1],    # Back-right
}


def forward_kinematics_leg(joint_angles, leg_idx, orientation=None, segment_length=0.1):
    """
    Compute foot position via forward kinematics with 3D body orientation.
    See docs/PHYSICS.md for modeling details (2D pitch-plane articulation + 3D rotation).
    
    Args:
        joint_angles: (3,) angles for hip, knee, ankle (cumulative pitch rotations)
        leg_idx: which leg (0=FL, 1=FR, 2=BL, 3=BR)
        orientation: (3,) [pitch, yaw, roll] body orientation or None for identity
        segment_length: length of each joint segment
    
    Returns:
        foot_pos: (3,) foot position in world frame
    """
    # Create hip offset on correct device/dtype (cheap conversion from list)
    hip_pos = torch.tensor(_HIP_OFFSETS_RAW[leg_idx], dtype=joint_angles.dtype, device=joint_angles.device)
    
    # Forward kinematics: each joint adds to position in local (pitch plane)
    theta1, theta2, theta3 = joint_angles[0], joint_angles[1], joint_angles[2]
    
    # Cumulative angles for each segment (pitch plane only)
    angle1 = theta1
    angle2 = theta1 + theta2
    angle3 = theta1 + theta2 + theta3
    
    # Use torch trig ops to preserve autograd and device/dtype
    c1, s1 = torch.cos(angle1), torch.sin(angle1)
    p1_local = torch.stack([torch.tensor(0., dtype=joint_angles.dtype, device=joint_angles.device),
                            -segment_length * c1,
                            -segment_length * s1])
    
    c2, s2 = torch.cos(angle2), torch.sin(angle2)
    p2_local = p1_local + torch.stack([torch.tensor(0., dtype=joint_angles.dtype, device=joint_angles.device),
                                       -segment_length * c2,
                                       -segment_length * s2])
    
    c3, s3 = torch.cos(angle3), torch.sin(angle3)
    p3_local = p2_local + torch.stack([torch.tensor(0., dtype=joint_angles.dtype, device=joint_angles.device),
                                       -segment_length * c3,
                                       -segment_length * s3])
    
    # Apply body orientation to rotate local coordinates to world frame
    if orientation is not None:
        p3_local = _rotate_point_by_euler(p3_local, orientation)
        hip_pos = _rotate_point_by_euler(hip_pos, orientation)
    
    return hip_pos + p3_local


def _rotate_point_by_euler(point, euler_angles):
    """
    Rotate a 3D point by Euler angles (pitch, yaw, roll).
    
    Args:
        point: (3,) point to rotate
        euler_angles: (3,) [pitch, yaw, roll] in radians
    
    Returns:
        rotated: (3,) rotated point
    """
    pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]
    dtype = point.dtype
    device = point.device
    
    # Rotation matrices for each axis
    # Pitch (rotation around X axis)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    Rx = torch.tensor([
        [1., 0., 0.],
        [0., cp, -sp],
        [0., sp, cp]
    ], dtype=dtype, device=device)
    
    # Yaw (rotation around Z axis)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    Rz = torch.tensor([
        [cy, -sy, 0.],
        [sy, cy, 0.],
        [0., 0., 1.]
    ], dtype=dtype, device=device)
    
    # Roll (rotation around Y axis)
    cr, sr = torch.cos(roll), torch.sin(roll)
    Ry = torch.tensor([
        [cr, 0., sr],
        [0., 1., 0.],
        [-sr, 0., cr]
    ], dtype=dtype, device=device)
    
    # Combined rotation: Rz * Ry * Rx (yaw -> roll -> pitch order)
    R = Rz @ Ry @ Rx
    return R @ point


def compute_center_of_mass(joint_angles, orientation=None, segment_length=0.1):
    """
    Compute center of mass position for quadruped. See docs/PHYSICS.md.
    
    Args:
        joint_angles: (12,) all joint angles [FL, FR, BL, BR]
        orientation: (3,) [pitch, yaw, roll] body orientation or None for identity
        segment_length: length of each joint segment
    
    Returns:
        com_pos: (3,) center of mass position relative to body center
    """
    device = joint_angles.device
    dtype = joint_angles.dtype
    
    body_mass = 1.0
    foot_mass = 0.2  # Each foot is 20% of body
    
    com = torch.zeros(3, dtype=dtype, device=device)
    
    # Add body COM
    com += body_mass * torch.zeros(3, dtype=dtype, device=device)
    
    # Add foot COMs
    total_mass = body_mass
    for leg_idx in range(4):
        foot_pos = forward_kinematics_leg(joint_angles[leg_idx*3:(leg_idx+1)*3], leg_idx, orientation, segment_length)
        com += foot_mass * foot_pos
        total_mass += foot_mass
    
    return com / total_mass


def compute_foot_positions(joint_angles, orientation=None, segment_length=0.1):
    """
    Compute all 4 foot positions.
    
    Args:
        joint_angles: (12,) all joint angles [FL, FR, BL, BR]
        orientation: (3,) [pitch, yaw, roll] body orientation in radians, or None for identity
        segment_length: length of each joint segment
    
    Returns:
        feet: (4, 3) foot positions
    """
    feet = []
    for leg_idx in range(4):
        foot_pos = forward_kinematics_leg(joint_angles[leg_idx*3:(leg_idx+1)*3], leg_idx, orientation, segment_length)
        feet.append(foot_pos)
    return torch.stack(feet)  # (4, 3)


@dataclasses.dataclass
class Creature:
    """Single entity with RL state - quadruped with 4 legs."""
    en_id: int
    pos: torch.Tensor        # (3,) [x, y, z] - center of mass position
    velocity: torch.Tensor  # (3,) [vx, vy, vz]
    orientation: torch.Tensor  # (3,) [pitch, yaw, roll] in radians
    rnn_state: Tuple[torch.Tensor, torch.Tensor]  # (h, c) each (1, hidden)
    
    # Quadruped leg system: 4 legs, 3 joints each (hip, knee, ankle) = 12 DOF
    joint_angles: torch.Tensor      # (12,) joint angles in radians
    joint_velocities: torch.Tensor  # (12,) joint angular velocities
    foot_contact: torch.Tensor      # (4,) binary contact state per foot [0,1]
    
    # Leg parameters (fixed)
    leg_length: float = 0.3  # Total leg length (3 segments of 0.1 each)
    segment_length: float = 0.1  # Length of each joint segment


class Encoder(nn.Module):
    """PATTERN A: Shared encoder trained ONLY by Dreamer on replay buffer.
    
    - Prevents representation drift from mixed on/off-policy objectives
    - PPO uses frozen (or EMA-synced) output from this encoder
    - Swap with CNN if obs are images.
    """
    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

    def forward(self, x):
        # x: (N, obs_dim)
        return self.net(x)  # (N, embed_dim)


class PPOPolicy(nn.Module):
    """PATTERN A: Policy controller using frozen Dreamer encoder.
    
    Architecture:
      1) Input: latent features from Dreamer encoder (frozen)
      2) Projection: Linear(latent_dim → lstm_hidden)
      3) LSTM: Per-entity temporal memory (unidirectional, causal for RL)
      4) Policy+Value heads: Output 12D motor torques and scalar value estimate
    
    Key: Encoder is frozen or updated via slow EMA from Dreamer training.
    This prevents representation drift from mixed on/off-policy objectives.
    """
    def __init__(
        self,
        latent_dim: int,        # Input dimension (from frozen encoder)
        lstm_hidden: int,
        num_actions: int = 12,
    ):
        super().__init__()

        # Input: latent features from shared encoder (NOT training this encoder)
        self.feature_projection = nn.Linear(latent_dim, lstm_hidden)

        self.lstm = nn.LSTMCell(lstm_hidden, lstm_hidden)

        # Policy head: outputs mean and std for 12D motor torques
        self.policy_mu = nn.Linear(lstm_hidden, num_actions)

        # ✅ STABILITY: log_std is a global learned parameter (not state-dependent)
        self.log_std_param = nn.Parameter(torch.zeros(num_actions))

        self.value = nn.Linear(lstm_hidden, 1)


class EntityBelief(nn.Module):
    """PATTERN A wrapper: Shared encoder + PPOPolicy
    
    Architecture:
      1) Encoder (SHARED): 34D obs → 128D features (trained ONLY by Dreamer on replay)
      2) PPOPolicy: LSTM + Policy/Value heads (uses frozen encoder output)
    
    This prevents representation drift from mixed on/off-policy training objectives.
    - Dreamer trains encoder offline on replay buffer
    - PPO uses frozen (or EMA-synced) encoder output
    - Only PPOPolicy is updated during PPO training
    
    Outputs 12D motor torques for quadruped legs.
    """
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        gat_out_dim: int,  # Unused, kept for config compatibility
        gat_heads: int,    # Unused, kept for config compatibility
        lstm_hidden: int,
        num_actions: int = 12,  # 12 joint torques for quadruped
    ):
        super().__init__()

        # SHARED: encoder trained only by Dreamer on replay buffer
        self.encoder = Encoder(obs_dim, embed_dim)
        
        # PPO policy uses frozen encoder output
        self.policy = PPOPolicy(
            latent_dim=embed_dim,
            lstm_hidden=lstm_hidden,
            num_actions=num_actions
        )
        
        # Freeze encoder for PPO (only Dreamer will train it)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def init_state(self, N: int, device, dtype):
        """Initialize LSTM state for N entities."""
        H = self.policy.lstm.hidden_size
        return (
            torch.zeros((N, H), device=device, dtype=dtype),
            torch.zeros((N, H), device=device, dtype=dtype),
        )

    def forward(
        self,
        obs_t: torch.Tensor,                       # (N, obs_dim)
        edge_index_t: torch.Tensor = None,         # Unused (kept for compatibility)
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (h, c) each (N, H)
    ):
        N = obs_t.size(0)
        device, dtype = obs_t.device, obs_t.dtype

        # 1) encode observations (FROZEN - trained only by Dreamer)
        with torch.no_grad():
            e_t = self.encoder(obs_t)              # (N, 128)

        # 2) project latent features to LSTM input dimension
        g_t = self.policy.feature_projection(e_t)  # (N, lstm_hidden)

        # 3) temporal memory per entity
        if prev_state is None:
            prev_state = self.init_state(N, device, dtype)

        h_t, c_t = self.policy.lstm(g_t, prev_state)  # (N, hidden)

        # 4) policy and value heads
        mu = self.policy.policy_mu(h_t)            # (N, A)
        
        # ✅ Global log_std (stable, not state-dependent)
        log_std = self.policy.log_std_param.expand_as(mu)  # (N, A)
        log_std = torch.clamp(log_std, -5, 2)      # PPO-safe range: std in [0.0067, 7.4]
        
        v = self.policy.value(h_t).squeeze(-1)     # (N,)

        return (mu, log_std), v, (h_t, c_t)


def init_single_creature(model, en_id=0, pos=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), device=None):
    """
    Initializes a single quadruped creature (4 legs, 3 joints each).

    Returns:
      creature: Creature dataclass with all leg/joint states
      edge_index: (2, E) graph edges
      
    Graph Structure:
      CURRENT (Optimized for vectorized training):
        - Single node (0) with self-loop
        - Edge: (0 -> 0), so edge_index = [[0], [0]]
        - Effectively just a linear projection (attention simplifies to identity for N=1)
      
      FUTURE (could implement for explicit joint modeling):
        - 12 nodes: one per joint (3 per leg × 4 legs)
        - Edges: kinematic chain (hip→knee→ankle per leg) + optional hip coordination
        - Node features: joint angles, velocities, previous actions
        - Would enable explicit learning of joint dependencies
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get dtype from model parameters
    dtype = next(model.parameters()).dtype

    # Creature tensors
    pos_t = torch.tensor(pos, dtype=dtype, device=device)
    vel_t = torch.zeros(3, dtype=dtype, device=device)
    ori_t = torch.tensor(orientation, dtype=dtype, device=device)

    # LSTM state for one entity (N=1)
    if hasattr(model, 'init_state'):
        h0, c0 = model.init_state(1, device, dtype)
    else:
        h_size = model.lstm.hidden_size
        h0 = torch.zeros((1, h_size), dtype=dtype, device=device)
        c0 = torch.zeros((1, h_size), dtype=dtype, device=device)

    # Quadruped leg system: 4 legs, 3 joints each = 12 DOF
    # Initialize joints in a neutral standing position
    # Legs: FL (0-2), FR (3-5), BL (6-8), BR (9-11)
    # Neutral standing: slightly flexed knees
    joint_angles = torch.tensor(
        [0.3, 0.6, 0.3,   # Front-left leg
         0.3, 0.6, 0.3,   # Front-right leg
         0.3, 0.6, 0.3,   # Back-left leg
         0.3, 0.6, 0.3],  # Back-right leg
        dtype=dtype, device=device
    )
    joint_velocities = torch.zeros(12, dtype=dtype, device=device)
    foot_contact = torch.ones(4, dtype=dtype, device=device)  # All feet on ground initially

    creature = Creature(
        en_id=en_id,
        pos=pos_t,
        velocity=vel_t,
        orientation=ori_t,
        rnn_state=(h0, c0),

        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        foot_contact=foot_contact,
    )

    # Graph structure: Currently a single-node self-loop (efficiency optimization)
    # For multi-node graph, would have 12 nodes with kinematic chain + hip coupling edges
    # See init_single_creature docstring for full 14-edge graph specification
    edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

    return creature, edge_index


class WorldModel(nn.Module):
    """DreamerV3-inspired world model for learning environment representation.
    
    PATTERN A ARCHITECTURE:
    - Uses SHARED encoder (same as PPO policy)
    - Encoder is trainable ONLY here, frozen elsewhere
    - Learns latent dynamics, reward prediction, termination on REPLAY BUFFER
    - Off-policy learning prevents drift from mixed on/off-policy objectives
    
    Trained on rollout buffer (old experiences), not on-policy sampled data.
    """
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 128, hidden_dim: int = 256, encoder: Encoder = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Use SHARED encoder if provided, otherwise create one
        if encoder is not None:
            self.encoder = encoder  # Will be shared with PPOPolicy
        else:
            self.encoder = Encoder(obs_dim, latent_dim)
        
        # Dynamics model: (latent_state, action) -> next_latent_state
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Reward predictor: latent_state -> reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Termination predictor: latent_state -> done probability
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Decoder: latent_state -> obs (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode(self, obs):
        """Encode observation to latent state (trains shared encoder)."""
        return self.encoder(obs)
    
    def decode(self, latent):
        """Decode latent state back to observation space."""
        return self.decoder(latent)
    
    def predict_next(self, latent, action):
        """Predict next latent state given current latent and action."""
        combined = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(combined)
        return next_latent
    
    def predict_reward(self, latent):
        """Predict reward from latent state."""
        return self.reward_head(latent)
    
    def predict_done(self, latent):
        """Predict termination probability from latent state."""
        return torch.sigmoid(self.done_head(latent))
    
    def forward(self, obs, action):
        """Full forward pass: obs -> encode -> dynamics -> reward/done/next_obs."""
        latent = self.encode(obs)
        next_latent = self.predict_next(latent, action)
        reward = self.predict_reward(next_latent)
        done_prob = self.predict_done(next_latent)
        next_obs_recon = self.decode(next_latent)
        return next_latent, reward, done_prob, next_obs_recon