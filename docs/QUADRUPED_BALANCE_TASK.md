# Quadruped Balance Task - Agent-Centered World

## Overview

This document describes the quadruped robot balance task with **improved physics** and **agent-centered world coordinates**. The goal is for the agent to maintain balance while positioning its **center of mass (COM)** as close to a target goal position as possible.

### Key Improvements

1. **Better Physics**: Gravity integration, contact force modeling, joint velocity limits
2. **Agent-Centered World**: Agent COM always at origin, world moves relative to agent
3. **Improved Collision Detection**: Foot-ground contact with spring-damper model

## Architecture Changes

## Architecture Changes

### Creature Structure (`entity.py`)

Previously: Simple point mass with position, velocity, orientation
Now: **Quadruped with 4 legs, 3 joints each (12 DOF)**

```python
@dataclass
class Creature:
    pos: torch.Tensor              # COM position (3,)
    velocity: torch.Tensor         # COM velocity (3,)
    orientation: torch.Tensor      # Pitch, yaw, roll (3,)
    
    # NEW: Quadruped leg system
    joint_angles: torch.Tensor     # (12,) - all joint angles
    joint_velocities: torch.Tensor # (12,) - all angular velocities
    foot_contact: torch.Tensor     # (4,) - contact state per foot
    
    leg_length: float = 0.3        # Total leg length
    segment_length: float = 0.1    # Per-joint segment length
```

### Leg Configuration

- **4 Legs**: Front-Left (FL), Front-Right (FR), Back-Left (BL), Back-Right (BR)
- **3 Joints per leg**: Hip, Knee, Ankle
- **Joint Index Mapping**:
  - FL: indices 0-2
  - FR: indices 3-5
  - BL: indices 6-8
  - BR: indices 9-11

### Forward Kinematics

For each leg, the foot position is computed using cumulative joint angles:

```python
def forward_kinematics_leg(joint_angles, leg_idx, segment_length=0.1):
    """
    Computes 3D foot position given 3 joint angles.
    Uses sequential joint additions in the vertical plane.
    """
    theta1, theta2, theta3 = joint_angles
    
    # Cumulative angles
    angle1 = theta1
    angle2 = theta1 + theta2
    angle3 = theta1 + theta2 + theta3
    
    # Segment positions (adding each joint contribution)
    p1 = hip_pos + [0, -L*cos(angle1), -L*sin(angle1)]
    p2 = p1 + [0, -L*cos(angle2), -L*sin(angle2)]
    p3 = p2 + [0, -L*cos(angle3), -L*sin(angle3)]
    
    return p3  # Foot position
```

### Center of Mass Calculation

COM is computed as weighted average of body and foot positions:

```python
def compute_center_of_mass(joint_angles, device, dtype, segment_length=0.1):
    body_mass = 1.0
    foot_mass = 0.2  # Each foot is 20% of body
    
    com = body_mass * body_pos + sum(foot_mass * foot_pos) / total_mass
    return com
```

## Physics Engine (`PhysicsEngine`)

### AGENT-CENTERED WORLD

The agent's center of mass is **always kept at the origin (0, 0, 0)** in the agent's local coordinate frame. The world coordinates are transformed relative to the agent.

**Benefits:**
- Simplifies observation: Goal position is naturally relative
- Reduces translation invariance problem
- More efficient for RL learning
- Matches how animals perceive their environment

```python
# Example: Agent is at global position (50, 20, 0.5)
# In agent-centered view:
agent.pos = [0, 0, 0]  # Always at origin
goal.pos_relative = goal.global_pos - agent.global_pos  # Relative coordinates
```

### Improved Gravity Integration

Gravity is applied based on **foot contact state**:

```python
# If all 4 feet are in contact, gravity is balanced by contact forces
# If only some feet are in contact, residual gravity applies
gravity_acceleration = gravity * (1.0 - num_contacts / 4.0)
velocity_z -= gravity_acceleration * dt
```

This allows:
- Natural standing without drift
- Realistic falling when balance is lost
- Smooth transitions between contact states

### Contact Force Modeling

Foot-ground contact uses a **spring-damper model**:

```python
def detect_contact(foot_z):
    if foot_z <= ground_level + foot_height_threshold:
        # Penetration depth
        penetration = max(0, ground_level - foot_z)
        
        # Contact force = spring restoring force + damping
        contact_force = contact_stiffness * penetration - contact_damping * velocity_z
        
        return contact_force
```

**Parameters:**
- `contact_stiffness = 0.5`: Ground resistance (higher = stiffer ground)
- `contact_damping = 0.2`: Friction/energy dissipation
- `foot_height_threshold = 0.05m`: Tolerance for contact detection

### Joint Dynamics with Velocity Clamping

Joint velocities are clamped to prevent instability:

```python
# Update velocity with torque and damping
velocity += (torque - joint_damping * velocity) * dt

# Prevent unrealistic joint speeds
velocity = clamp(velocity, -10 rad/s, +10 rad/s)
```

### Action Space

**Changed from 3D acceleration → 12D motor torques**

Each joint receives a target torque:

```python
# Raw action from policy: (1, 12) tanh output in [-1, 1]
motor_torques = action * 5.0  # Scaled to [-5, 5] N⋅m

# Clamped for stability
motor_torques = clamp(motor_torques, -max_torque, +max_torque)
```

### Reward Function - **BALANCE FOCUSED**

Replaced navigation-based rewards with balance-based rewards:

```python
def compute_balance_reward(creature, com_pos, stability_metrics, motor_torques, goal_pos):
    # 1. COM Distance Reward (main objective)
    com_xy = [com_pos[0], com_pos[2]]  # XZ plane
    com_dist = ||com_xy - goal_xy||
    
    if com_dist < threshold:
        com_reward = (threshold - com_dist) * reward_scale
    else:
        com_reward = -com_dist * 0.5
    
    # 2. Stability Reward
    pitch, roll = creature.orientation[:2]
    tilt_penalty = -5.0 if (|pitch| > 0.5 or |roll| > 0.5) else 0
    
    # 3. Contact Reward (prefer multiple feet on ground)
    num_contacts = sum(foot_contact > 0.5)
    contact_reward = min(num_contacts, 4) * contact_reward_scale
    
    # 4. Energy Efficiency
    energy_cost = ||motor_torques|| * energy_penalty
    
    # Total
    total_reward = com_reward + tilt_penalty + contact_reward - energy_cost
    return total_reward
```

## Observation Space (`Environment.observe`)

**37D observation in agent-centered coordinates**

```
12D: Joint angles (θ₁₋₁₂)
12D: Joint velocities (ω₁₋₁₂)
 4D: Foot contact states (c₁₋₄) ∈ [0, 1]
 3D: Body orientation (pitch, yaw, roll)
 3D: COM position (always ~0 in agent-centered frame)
 3D: Goal position relative to COM
-----
37D: Total observation
```

### Agent-Centered Observation

The observation is constructed in the **agent's local reference frame**:

```python
def observe(creature):
    obs = concat([
        creature.joint_angles,                          # (12,)
        creature.joint_velocities,                      # (12,)
        creature.foot_contact,                          # (4,)
        creature.orientation,                           # (3,)
        zeros(3),  # COM is at origin in local frame   # (3,)
        goal_pos_global - creature.pos_global           # (3,) relative goal
    ])
    return obs.unsqueeze(0)  # (1, 37)
```

**Why agent-centered?**
- Goal position naturally becomes relative coordinates
- Agent always perceives itself at the center
- Simplifies temporal consistency in policy
- Matches proprioceptive sensing in animals

### Foot Contact Representation

Foot contact is a **continuous value in [0, 1]** representing the depth of contact:

```python
contact_state = 1.0 if foot_z <= ground_level else 0.0
# Allows smooth transitions during liftoff/touchdown
```

## Neural Network Updates

### EntityBelief

```python
class EntityBelief(nn.Module):
    def __init__(
        obs_dim=37,          # 37D quadruped observation
        num_actions=12,      # 12D motor torques (default, changed from 3)
        ...
    ):
        # Outputs μ and log_std for 12D Gaussian policy
        self.policy_mu = nn.Linear(lstm_hidden, 12)
        self.log_std_param = nn.Parameter(torch.zeros(12))
```

### Action Sampling (12D)

Log probability calculation updated for 12D space:

```python
# In collect_trajectory:
log_prob_gaussian = -0.5 * ((u - mu) ** 2 / (std ** 2)).sum(dim=1)
log_prob_gaussian -= log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI  # 12 dimensions
tanh_correction = -torch.log(1.0 - action_batch ** 2).sum(dim=1)  # 12D sum
log_prob = log_prob_gaussian + tanh_correction
```

## Configuration Updates (`config.py`)

Key changes:

```python
# Observation/Action dimensions
OBS_DIM = 37  # From 7
ACTION_DIM = 12  # From 3

# Memory model updated
OBS_BYTES = 148  # (1, 37) float32
PER_STEP_BYTES = 300  # Larger action space

# Balance task rewards (replaces navigation rewards)
COM_DISTANCE_THRESHOLD = 0.3
COM_REWARD_SCALE = 10.0
STABILITY_REWARD_SCALE = 1.0
CONTACT_REWARD = 0.1
ENERGY_PENALTY = 0.01

# Quadruped physics
JOINT_DAMPING = 0.01
MAX_TORQUE = 5.0
SEGMENT_LENGTH = 0.1
GRAVITY = 9.8  # Actual gravity (not scaled)
```

## Training Changes

### Episode Initialization (Agent-Centered)

```python
# Reset to standing position with neutral joint angles
# In agent-centered world, starting position is always at origin
creature.pos = [0.0, 0.0, 0.5]  # Will be normalized to origin
creature.joint_angles = [0.3, 0.6, 0.3] * 4  # Neutral standing
creature.joint_velocities = zeros(12)
creature.foot_contact = ones(4)  # All feet on ground initially
```

### Goal Spawning (Agent-Centered)

Goals are spawned **relative to the agent** within reachable distance:

```python
# Goal is placed 0.3-1.0m away from agent COM
# In agent-centered frame: goal is offset from origin
goal_offset = random_direction() * random_distance(0.3, 1.0)
goal.pos = agent.pos + goal_offset  # In global frame
```

### Termination Condition

Changed from goal distance to COM distance (in agent-centered frame):

```python
com_distance = ||com_xy - goal_xy||  # Both relative to agent
done = com_distance < com_distance_threshold  # 0.3m
```

### Episode Reset Flow

1. **Global Position**: Agent placed at origin in agent-centered world
2. **Local Observation**: All observations computed relative to agent
3. **Goal Placement**: Random offset from agent
4. **Training**: Network learns in agent-centered frame
5. **Generalization**: Works for any global position (translation invariant)

## Testing Checklist

- [ ] Creature initialization with quadruped state
- [ ] Forward kinematics produces valid foot positions
- [ ] COM calculation is physically reasonable
- [ ] Motor torques correctly update joint states
- [ ] Observation tensor has correct shape (1, 37)
- [ ] Policy outputs correct action shape (1, 12)
- [ ] Reward is computed from COM position, not goal distance
- [ ] Training loop converges (agent learns to balance)
- [ ] Evaluation shows stable quadruped stance

## Future Enhancements

1. **Locomotion**: Add forward/backward movement rewards to enable walking
2. **Terrain**: Add uneven ground, obstacles
3. **Dynamics Randomization**: Vary gravity, inertia, friction for robustness
4. **Gait Learning**: Learn different walking gaits (trot, gallop, etc.)
5. **Inverse Kinematics**: Plan foot trajectories directly instead of joint angles
