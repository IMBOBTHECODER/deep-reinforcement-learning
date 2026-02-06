# Agent System - Quadruped Motor Control & Learning

Comprehensive documentation of the quadruped agent architecture, neural networks, motor control, and learning mechanisms for balance task.

## Agent Overview

The agent is a 4-legged quadruped with 12 degrees of freedom (3 joints per leg) that learns to balance and move toward goals efficiently. It combines:

1. **Policy Network** - Generates 12D motor torques (actions) for joint control
2. **Value Network** - Estimates expected future reward from current state
3. **Motor Control System** - Translates motor torques to joint dynamics

The quadruped learns through deep reinforcement learning (PPO) to maintain balance while moving toward goal positions in an agent-centered coordinate frame.

## Agent Architecture

### Quadruped Morphology

```
         Front
    FL    |    FR
     |\   |   /|
     | \  |  / |   Front-Left (FL): Leg 0
     |  Body  |    Front-Right (FR): Leg 1
     | /  |  \ |    Back-Left (BL): Leg 2
     |/   |   \|    Back-Right (BR): Leg 3
    BL    |    BR
         Back
```

**Specifications:**
- **Legs**: 4 (Front-Left, Front-Right, Back-Left, Back-Right)
- **Joints per leg**: 3 (hip, knee, ankle in pitch axis)
- **Total DOF**: 12 (3 joints × 4 legs)
- **Segment length**: 0.1m per joint
- **Action type**: Motor torques (continuous control)
- **Action range**: [-5, 5] N⋅m per motor

#### 2. Feature Encoder
Extracts meaningful patterns from agent state and observations:
- **Input**: 37D observations (agent-centered coordinates)
  - 12D: Joint angles [θ₁-θ₁₂]
  - 12D: Joint velocities [ω₁-ω₁₂]
  - 4D: Foot contact states [c₁-c₄] (continuous, 0-1)
  - 3D: Body orientation [roll, pitch, yaw]
  - 3D: COM position (zeros in agent-centered frame)
  - 3D: Goal relative position [Δx, Δy, Δz]
- **Processing**: 2-layer MLP
  - Input: 37 → Hidden: 256 → Hidden: 256
  - Output: 128-dim encoded features
  - Activation: ReLU between layers, none at output

```python
# In entity.py (EntityBelief class)
self.fc1 = nn.Linear(37, 256)    # 37D observation
self.fc2 = nn.Linear(256, 256)
self.fc3 = nn.Linear(256, 128)   # 128D embedding
```

#### 3. LSTM (Long Short-Term Memory)
Maintains temporal state for sequential decision-making:
- **Hidden Size**: 256 units (increased for complex motor control)
- **Layers**: 1 bidirectional
- **Purpose**: Remember previous actions and states, learn smooth motion sequences
- **Input**: Encoded features (128D) + previous action (12D)
- **Output**: LSTM hidden state (256D used for policy/value heads)

```python
# LSTM state tracking
self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
rnn_state_new = lstm(features, rnn_state_old)
# Output: (batch_size, seq_len, 256)
```

#### 4. Policy Head - 12D Motor Torque Generation
Generates motor control signals for all 12 joints:
- **Input**: LSTM hidden state (256D)
- **Output**: Mean torques (μ) for 12 joints
- **Activation**: Tanh to bound actions to [-1, 1]
- **Range after scaling**: [-5, 5] N⋅m per motor (in PhysicsEngine)
- **Distribution**: Gaussian with learned per-action standard deviation
- **Benefit**: Continuous differentiable control

```python
# Policy head computation
self.policy_head = nn.Linear(256, 12)   # 256D → 12D motor torques
action_mean = torch.tanh(self.policy_head(rnn_state))  # [-1, 1]
# Scaled to [-5, 5] N*m in training loop
action_noise = torch.randn_like(action_mean)
action = action_mean + exp(log_std) * action_noise  # Reparameterization
```

**Motor Mapping:**
- Torques [0-2]: Front-Left leg (hip, knee, ankle)
- Torques [3-5]: Front-Right leg
- Torques [6-8]: Back-Left leg
- Torques [9-11]: Back-Right leg

#### 5. Value Head - Balance Potential Estimation
Estimates expected cumulative reward (value function):
- **Input**: LSTM hidden state (256D)
- **Output**: Scalar value estimate V(s)
- **Range**: Typically -5 to +200 (based on reward scale)
- **Purpose**: Advantage computation for PPO training
- **Loss**: MSE between predicted and actual returns

```python
# Value head computation
self.value_head = nn.Linear(256, 1)  # 256D → scalar
value = self.value_head(rnn_state)   # Unbounded output
```

### Neural Network Full Architecture

```
INPUT OBSERVATION (37D)
    ↓
┌───────────────────────────┐
│  Feature Encoder (2 MLP)  │
│   37 → 256 → 256 → 128   │
│  (Learned feature extraction)
└────────────┬────────────┘
             ↓
        [128D embedding]
             ↓
┌───────────────────────────┐
│    LSTM Layer (256)       │
│  Processes sequence of   │
│  features + prev actions │
│  (Temporal dependency)    │
└────────────┬────────────┘
             ↓
        [256D hidden state]
         ↙          ↘
    POLICY HEAD   VALUE HEAD
     (12D out)     (1D out)
         ↓              ↓
    Motor Torques   Value Est.
    [-5, 5] N*m    Cumulative Reward
```

## Motor Control & Joint Dynamics

### Motor Torque Application

Motor torques control joint velocities through a first-order dynamics model:

**Joint Dynamics Equation:**
```
ω_{t+1} = ω_t + (τ - d·ω_t) · Δt

Where:
  ω = joint angular velocity (rad/s)
  τ = motor torque (N·m)
  d = damping coefficient
  Δt = timestep (0.02s at 50Hz)
```

**Then angles update:**
```
θ_{t+1} = θ_t + ω_{t+1} · Δt
```

### Joint Velocity Clamping

To maintain stability and prevent unrealistic joint speeds:

```python
# In PhysicsEngine.apply_motor_torques()
max_joint_vel = 10.0  # ±10 rad/s (~600 RPM)
creature.joint_velocities = clamp(creature.joint_velocities, -max_joint_vel, max_joint_vel)
```

**Benefits:**
- Prevents explosive joint accelerations
- Matches real motor constraints
- Stabilizes learning

### Gravity Integration with Contact State

Gravity is applied conditionally based on foot contact:

```python
# Contact-dependent gravity
num_contacts = sum(foot_contact)  # 0-4 feet
gravity_factor = 1.0 - num_contacts / 4.0

# Apply gravity
creature.velocity.z -= GRAVITY * gravity_factor * dt
```

**Physics:**
- **4 feet in contact** (standing): gravity_factor = 0.0 → no downward acceleration
- **2 feet in contact** (walking): gravity_factor = 0.5 → 50% gravity
- **0 feet in contact** (falling): gravity_factor = 1.0 → full gravity (9.8 m/s²)

This enables natural balance learning without explicit balance rewards.

## Observation Space (37D, Agent-Centered)

The agent perceives the world in agent-centered coordinates (agent always at origin):

| Index | Dimension | Name | Range | Meaning |
|-------|-----------|------|-------|---------|
| 0-11 | 12D | `joint_angles` | [-π, π] | Angles for all 12 joints |
| 12-23 | 12D | `joint_velocities` | [-10, 10] | Velocities for all 12 joints (rad/s) |
| 24-27 | 4D | `foot_contacts` | [0, 1] | Contact state per foot (continuous) |
| 28-30 | 3D | `orientation` | [-π, π] | Body roll, pitch, yaw |
| 31-33 | 3D | `com_position` | [0, 0, 0] | Center of mass (always zero in agent frame) |
| 34-36 | 3D | `goal_relative` | [-∞, ∞] | Goal position relative to agent (Δx, Δy, Δz) |

**Agent-Centered Coordinates:**
All observations are automatically transformed to agent-local frame:
- Agent COM is always at (0, 0, 0)
- World coordinates are relative to agent
- Goal is at goal_global - agent_global
- Benefits: Translation invariance, generalizes to any environment

### Observation Normalization

```python
# In normalize_obs()
normalized = (obs - obs_mean) / (obs_std + 1e-8)
```

Normalization is learned online to keep values in reasonable range.

## Action Space (12D Motor Torques)

Motor torques control all 12 joints directly:

| Index | Joint | Range | Meaning |
|-------|-------|-------|---------|
| 0-2 | FL hip, knee, ankle | [-5, 5] N·m | Front-left leg |
| 3-5 | FR hip, knee, ankle | [-5, 5] N·m | Front-right leg |
| 6-8 | BL hip, knee, ankle | [-5, 5] N·m | Back-left leg |
| 9-11 | BR hip, knee, ankle | [-5, 5] N·m | Back-right leg |

**Action Generation:**
```python
# Policy network outputs tanh-bounded actions
action_logits = policy_head(rnn_state)  # (batch, 12)
action_tanh = torch.tanh(action_logits)  # (batch, 12) in [-1, 1]

# Scale to motor torque range
action_scaled = action_tanh * MAX_TORQUE  # (batch, 12) in [-5, 5] N·m

# Add exploration noise during training
log_std = torch.tensor([...]  # Learned per-action
action_sampled = action_scaled + exp(log_std) * randn_like(action_scaled)
```

**Log Probability Calculation:**
```python
# Gaussian log probability for 12D action
log_prob = -0.5 * (((action - mu) / sigma) ** 2).sum(dim=-1)
log_prob -= 0.5 * 12 * math.log(2 * math.pi)  # 12D Gaussian term
log_prob -= log_std.sum()  # Jacobian for tanh transformation
```

## Learning Process - PPO (Proximal Policy Optimization)

### Data Collection Phase

For each environment:
1. Agent observes state s_t (37D)
2. Policy generates action a_t (12D motor torques)
3. PhysicsEngine executes action, returns:
   - next_state s_{t+1}
   - reward r_t
   - done flag
   - log_prob(a_t | s_t)
4. Value network estimates V(s_t)
5. Collect trajectory: {s, a, r, v, log_prob}

```python
# In collect_trajectory()
for step in range(MAX_STEPS_PER_EPISODE):
    obs = env.observe()  # (1, 37)
    action_mean, action_logstd = policy.forward(obs)
    action = action_mean + exp(action_logstd) * randn_like(action_mean)
    
    next_obs, reward, done = env.step(action)
    value = value_net(obs)
    
    trajectory.append((obs, action, reward, value, log_prob))
```

### Training Phase (4 PPO Epochs)

For each collected trajectory:

**1. Advantage Computation (GAE):**
```python
# Generalized Advantage Estimation
advantages = []
gae = 0
for step in reversed(range(T)):
    delta = reward[step] + GAMMA * value[step+1] - value[step]
    gae = delta + GAMMA * GAE_LAMBDA * gae
    advantages.insert(0, gae)
```

**2. Policy Loss (Clipped):**
```python
# PPO clipped objective
ratio = exp(log_prob_new - log_prob_old)
surrogate1 = ratio * advantages
surrogate2 = clamp(ratio, 1-CLIP, 1+CLIP) * advantages
policy_loss = -min(surrogate1, surrogate2).mean()
```

**3. Value Loss:**
```python
# Temporal difference learning
targets = advantages + values_old
value_loss = ((value_new - targets) ** 2).mean()
```

**4. Entropy Bonus:**
```python
# Encourage exploration
entropy = -log_prob.mean()
total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
```

**5. Backward Pass:**
```python
total_loss.backward()
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
optimizer.step()
```

## Learning Dynamics

### What the Agent Learns

**Early episodes (1-20):**
- Random exploration phase
- Learns basic joint movements
- High motor noise, jerky movements

**Middle episodes (20-80):**
- Discovers balance principle (keep COM low)
- Learns contact-gravity relationship
- Smoother movements, emergent gaits

**Late episodes (80+):**
- Optimizes balance-to-goal tradeoff
- Learns efficient walking patterns
- Smooth, coordinated leg movements
- Reaches goals with minimal energy

### Training Stability Features

1. **Gradient Clipping**: Prevents exploding gradients
   ```python
   clip_grad_norm_(parameters, max_norm=1.0)
   ```

2. **Advantage Normalization**: Stabilizes value estimation
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

3. **Learning Rate Scheduling**: Optional warmup then decay
   ```python
   lr = initial_lr * (1 - progress)  # Linear decay
   ```

## Agent Capabilities at Convergence

Once trained (100+ episodes):
- **Balance**: Maintains upright stance on flat ground
- **Walking**: Smooth gaits (trot-like pattern for quadrupeds)
- **Goal-reaching**: Walks toward goals with 80%+ success
- **Efficiency**: Minimizes motor torque while moving
- **Robustness**: Generalizes to different goal positions (agent-centered coords)
real_movement = tanh(network_output) * MAX_SPEED
```

## Stamina System

### Design Philosophy

Stamina transforms basic navigation into strategic resource management:
- Agent must choose between speed (high stamina) and efficiency (conservation)
- Requires forward planning: "Do I have enough energy for this path?"
- Creates natural curriculum: Early episodes focus on just reaching goals, later episodes optimize energy

### Stamina Mechanics

#### Parameters
```python
MAX_STAMINA = 200.0          # Full energy per episode
WALK_COST = 0.5              # Cost per normal movement step
JUMP_COST = 1.0              # Cost for high Z movement (|dz| > 0.1)
STAMINA_REGEN = 1.0          # Regeneration when idle
JUMP_THRESHOLD = 0.1         # Z threshold for jump classification
```

#### Cost Calculation
```python
def calculate_stamina_cost(dx, dy, dz, is_moving):
    movement_mag = sqrt(dx² + dy² + dz²)
    
    if not is_moving or movement_mag < 0.01:
        return 0.0  # Standing still costs nothing
    
    # Cost scales linearly with movement magnitude
    if abs(dz) > JUMP_THRESHOLD:
        return movement_mag * JUMP_COST  # 1.0 per unit vertical movement
    else:
        return movement_mag * WALK_COST  # 0.5 per unit horizontal movement
```

#### Depletion Mechanics
```python
creature.stamina -= cost              # Subtract movement cost
creature.stamina += STAMINA_REGEN     # Add regeneration if idle
creature.stamina = clamp(0, MAX_STAMINA)  # Enforce bounds

# Check for depletion
if creature.stamina < 0.1:
    # Severe penalty applied to reward
    reward -= STAMINA_DEPLETION_PENALTY  # -1.0
```

### Strategic Implications

#### Energy Planning
**Efficient Path**: 50 steps (25 energy cost, long time)
```
start → ... → ... → goal
Stamina needed: 25.0
Time: 50 steps
```

**Direct Path**: 10 jumps (10.0 energy cost, fast)
```
start → ↑ → ↑ → goal
Stamina needed: 10.0
Time: 10 steps
```

#### Learning Progression

**Episode 1-100** (Exploration)
- Agent learns basic navigation
- Stamina not yet limiting
- Reaches goals, wastes energy

**Episode 100-500** (Awareness)
- Agent starts running out of stamina on distant goals
- Learns to rest strategically
- Discovers efficient movement

**Episode 500+** (Optimization)
- Agent balances speed vs efficiency
- Makes intelligent energy tradeoffs
- Adapts strategy to goal distance

### Stamina in Observations

Stamina normalized to [0, 1] range in observation space:
```python
stamina_normalized = creature.stamina / MAX_STAMINA

# Enables agent to learn:
# - "I'm at 0.3 stamina, need to rest"
# - "I'm at 0.9 stamina, can be aggressive"
# - "This path requires 0.4 stamina, I have 0.7, safe"
```

## Agent Learning Dynamics

### Policy Gradient (PPO)

The agent learns through Policy Gradient Optimization:

1. **Collect Experience**: Run agent in environment, record trajectories
2. **Compute Advantages**: Estimate which actions were better than expected
3. **Update Policy**: Increase probability of good actions, decrease bad ones
4. **Clipping**: Limit update size to prevent collapse

```python
# PPO clipping
surrogate_loss = min(
    ratio * advantages,
    clip(ratio, 1-ε, 1+ε) * advantages
)
```

### Advantage Computation

Advantages measure "how good was this action compared to expectation":

```python
advantage = return - value_estimate

# Returns computed from rewards
return = discount_sum(rewards, gamma)

# Value estimates the return expectation
value_estimate = value_network(state)
```

**Examples**:
- Return = +10, Value = +5 → Advantage = +5 (good action!)
- Return = +2, Value = +5 → Advantage = -3 (bad action)

### Entropy Regularization

Encourages exploration by penalizing overly deterministic behavior:

```python
entropy_bonus = -entropy(policy_distribution)
loss = policy_loss + value_loss + entropy_coefficient * entropy_bonus
```

**Effect**:
- Early training: High entropy, agent explores broadly
- Later training: Entropy naturally decreases, policy focuses
- Benefit: Prevents premature convergence to local optima

## Agent State Management

### Creature Dataclass

```python
@dataclass
class Creature:
    pos: torch.Tensor           # Position [x, y, z]
    orientation: torch.Tensor   # Rotation quaternion
    rnn_state: Optional[Tuple]  # LSTM state for temporal memory
    stamina: torch.Tensor       # Current energy (0-200)
```

### State Persistence

Agent maintains LSTM hidden state across steps:
```python
# Step 1: Action 1, RNN state → State 1'
# Step 2: Action 2, RNN state → State 2'
# Step 3: Action 3, RNN state → State 3'
```

Enables learning of sequences:
- "After moving forward twice, turn left"
- "When low on stamina, rest"
- "If blocked, try alternate route"

### Episode Reset

At episode start:
```python
creature.pos = random_position()
creature.stamina = MAX_STAMINA
creature.rnn_state = None  # Initialize fresh RNN state
```

## Observation Processing Pipeline

```
Raw 7D Obs
    ↓
[Encoder: 7→256→256→128]
    ↓
128-dim Features
    ↓
[GAT: 4-head attention]
    ↓
128-dim Attended Features
    ↓
[LSTM: Process temporal]
    ↓
128-dim Hidden State
    ↓
    ├→ [Policy Head] → μ_action
    ├→ [Value Head] → V(state)
    └→ [Action Sample] → action
```

## Agent Capabilities & Limitations

### Capabilities
✅ Continuous movement in 3D
✅ Long-horizon planning (LSTM)
✅ Spatial reasoning (GAT)
✅ Energy-aware decision-making
✅ Multi-goal navigation
✅ Obstacle avoidance

### Limitations
❌ Single agent (no multi-agent coordination)
❌ Static environment (no dynamic obstacles)
❌ Visual input (input is position-based, not image)
❌ Communication (agents don't share information)

## Debugging Agent Behavior

### Check Agent State

```python
creature = system.creatures[0]
print(f"Position: {creature.pos}")
print(f"Stamina: {creature.stamina:.1f}/{system.max_stamina}")
print(f"RNN State: {creature.rnn_state[0].shape if creature.rnn_state else 'None'}")
```

### Monitor Learning

```python
# From training.log
episode 100 | reward: 25.3 | entropy: 0.82 | policy_loss: 0.12 | value_loss: 0.45
```

**Healthy signs**:
- Reward increasing over time
- Entropy decreasing gradually
- Policy loss staying around 0.1-0.5
- Value loss decreasing

### Visualize Agent Decision

```python
obs = system.observe(0)
print(f"Agent sees: {obs}")  # [rel_x, rel_y, rel_z, abs_x, abs_y, abs_z, stamina]

# Feed through network
with torch.no_grad():
    action, value = system.belief(obs, creature.rnn_state)
print(f"Action: {action}")     # Proposed movement
print(f"Value: {value:.2f}")   # Expected return
```

## Advanced: Custom Agent Modifications

### Adding Sensor Fusion

Extend 7D observations to include additional inputs:

```python
def observe_extended(agent_id):
    base_obs = system.observe(agent_id)  # 7D
    
    # Add more sensors
    goal_distance = distance(pos, goal)
    wall_proximity = min_distance_to_walls(pos)
    energy_efficiency = goals_completed / stamina_used
    
    extended_obs = torch.cat([
        base_obs,
        torch.tensor([goal_distance, wall_proximity, energy_efficiency])
    ])
    return extended_obs  # 10D now
```

### Custom Action Scaling

Change how action values map to movement:

```python
def apply_action_custom(action, stamina):
    # Conservative when low on energy
    energy_factor = stamina / MAX_STAMINA
    scaled_action = action * energy_factor
    return scaled_action
```

### Imitation Learning

Pre-train agent on expert demonstrations:

```python
def compute_imitation_loss(agent_output, expert_action):
    # Minimize KL divergence from expert policy
    loss = kl_divergence(agent_output, expert_action)
    return loss

# Add to training
loss = ppo_loss + imitation_loss
```

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System-level design
- [WORLD_SYSTEM.md](WORLD_SYSTEM.md) - Environment and rewards
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - How agents learn
- [CONFIGURATION.md](CONFIGURATION.md) - Agent parameters
