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
    """Per-entity feature encoder. Swap with CNN if obs are images."""
    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        # x: (N, obs_dim)
        return self.net(x)  # (N, embed_dim)


class SimpleGATLayer(nn.Module):
    """
    Multi-head graph attention (GAT) layer without torch_geometric.
    edge_index: (2, E) with source -> target edges.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.heads_out_dim = heads * out_dim

        # Linear projection per head: (in_dim) -> (heads*out_dim)
        self.W = nn.Linear(in_dim, self.heads_out_dim, bias=False)

        # Attention vectors per head: a^T [Wh_i || Wh_j]
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x: (N, in_dim)
        edge_index: (2, E) with edge_index[0]=src, edge_index[1]=dst
        returns:
          (N, heads*out_dim) if concat else (N, out_dim)
        """
        N = x.size(0)
        
        # ✅ OPTIMIZATION: For single node, attention is identity
        # This is a huge speedup in single-agent training
        if N == 1:
            h = self.W(x)  # (1, heads*out_dim)
            return h if self.concat else h.view(1, self.heads, self.out_dim).mean(dim=1)
        
        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)
        device = x.device
        dtype = x.dtype  # ✅ Match input dtype

        # Project node features
        h = self.W(x).view(N, self.heads, self.out_dim)  # (N, H, D)

        # Compute unnormalized attention scores e_ij per edge and head
        # e_ij = LeakyReLU( a_src·h_i + a_dst·h_j )
        e = (h[src] * self.a_src).sum(-1) + (h[dst] * self.a_dst).sum(-1)  # (E, H)
        e = self.leaky_relu(e)

        # Softmax over incoming edges per dst node, separately per head
        # ✅ FIX: Match dtype for numerical stability in mixed precision
        alpha = torch.zeros((src.size(0), self.heads), device=device, dtype=dtype)
        for head in range(self.heads):
            e_h = e[:, head]
            # Use proper negative infinity for dtype
            max_per_dst = torch.full((N,), torch.finfo(dtype).min, device=device, dtype=dtype)
            max_per_dst.scatter_reduce_(0, dst, e_h, reduce="amax", include_self=True)
            exp = torch.exp(e_h - max_per_dst[dst])
            denom = torch.zeros((N,), device=device, dtype=dtype)
            denom.scatter_add_(0, dst, exp)
            alpha[:, head] = exp / (denom[dst] + 1e-9)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: sum_j alpha_ij * h_j into dst i
        out = torch.zeros((N, self.heads, self.out_dim), device=device, dtype=dtype)
        for head in range(self.heads):
            msg = h[src, head, :] * alpha[:, head].unsqueeze(-1)
            out[:, head, :].scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        if self.concat:
            return out.reshape(N, self.heads_out_dim)
        else:
            return out.mean(dim=1)


class EntityBelief(nn.Module):
    """GAT at each timestep, then per-entity LSTM across time.
    Outputs 12D motor torques for quadruped legs."""
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        gat_out_dim: int,
        gat_heads: int,
        lstm_hidden: int,
        num_actions: int = 12,  # 12 joint torques for quadruped
    ):
        super().__init__()

        self.encoder = Encoder(obs_dim, embed_dim)
        self.gat = SimpleGATLayer(embed_dim, gat_out_dim, heads=gat_heads, concat=True, dropout=0.1)

        self.gat_feat_dim = gat_out_dim * gat_heads
        self.lstm = nn.LSTMCell(self.gat_feat_dim, lstm_hidden)

        # Policy head: outputs mean and std for 12D motor torques
        self.policy_mu = nn.Linear(lstm_hidden, num_actions)

        # ✅ STABILITY: log_std is a global learned parameter (not state-dependent)
        # This is the PPO baseline choice: prevents variance from exploding or collapsing
        self.log_std_param = nn.Parameter(torch.zeros(num_actions))

        self.value = nn.Linear(lstm_hidden, 1)

    def init_state(self, N: int, device, dtype):
        """Initialize LSTM state for N entities."""
        H = self.lstm.hidden_size
        return (
            torch.zeros((N, H), device=device, dtype=dtype),
            torch.zeros((N, H), device=device, dtype=dtype),
        )

    def forward(
        self,
        obs_t: torch.Tensor,                       # (N, obs_dim)
        edge_index_t: torch.Tensor,                # (2, E)
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (h, c) each (N, H)
    ):
        N = obs_t.size(0)
        device, dtype = obs_t.device, obs_t.dtype

        # 1) encode per entity
        e_t = self.encoder(obs_t)                  # (N, embed_dim)

        # 2) relational mixing at same timestep
        g_t = self.gat(e_t, edge_index_t)          # (N, gat_heads*gat_out_dim)

        # 3) temporal memory per entity
        if prev_state is None:
            prev_state = self.init_state(N, device, dtype)

        h_t, c_t = self.lstm(g_t, prev_state)      # (N, hidden)

        # 4) policy and value heads
        mu = self.policy_mu(h_t)                   # (N, A)
        
        # ✅ Global log_std (stable, not state-dependent)
        log_std = self.log_std_param.expand_as(mu)  # (N, A)
        log_std = torch.clamp(log_std, -5, 2)       # PPO-safe range: std in [0.0067, 7.4]
        
        v = self.value(h_t).squeeze(-1)            # (N,)

        return (mu, log_std), v, (h_t, c_t)


def init_single_creature(model, en_id=0, pos=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), device=None):
    """
    Initializes a single quadruped creature (4 legs, 3 joints each).

    Returns:
      creature: Creature dataclass with all leg/joint states
      edge_index: (2, 1) self-loop graph for GAT
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

    # Self-loop edge for a single node graph
    edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

    return creature, edge_index


class WorldModel(nn.Module):
    """
    DreamerV3-inspired world model for imagining future trajectories.
    Learns latent state transitions and reward prediction.
    """
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: obs -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
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
        """Encode observation to latent state."""
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