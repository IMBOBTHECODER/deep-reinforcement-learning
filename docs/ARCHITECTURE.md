# Architecture - Modular 5-Class Design

Comprehensive guide to the system's architecture, components, and data flow for the quadruped balance task.

## System Overview (Updated Feb 2026)

### Multi-Core Physics Optimization (NEW)

The system now distributes physics computation across CPU cores using ThreadPoolExecutor:

```
Neural Network (GPU) ────────┐
                             ├─ Data Collection Loop
Physics Computation (CPU) ───┤
  ├─ Thread 1: Reward calc   │
  ├─ Thread 2: Reward calc   ├─→ Parallel Processing
  ├─ Thread N: Reward calc   │
  └─ Main thread: NN inference
```

Benefits:
- **GPU**: Continuously processes actions (no idle time waiting for physics)
- **CPU**: Physics rewards computed in parallel across cores
- **Result**: Higher GPU utilization (60-80% vs 30-40%), faster training

### 5-Class Modular Architecture

```
┌──────────────────────────────────────────────────────┐
│     Quadruped Balance Training System                │
│         (5-Class Modular Design)                     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              SYSTEM (Orchestrator)                   │
│  ├─ Manages all 5 components                        │
│  ├─ Coordinates training loop                       │
│  └─ Handles checkpointing                           │
└─────┬────────────────────────────────────────────────┘
      │
      ├─→ ┌──────────────────────────────────────────┐
      │   │ ENVIRONMENT                              │
      │   │ ├─ World state, creatures, goals        │
      │   │ ├─ observe() → 37D obs                  │
      │   │ ├─ reset() → spawn quadruped            │
      │   │ └─ step() → apply actions               │
      │   └──────────────────────────────────────────┘
      │
      ├─→ ┌──────────────────────────────────────────┐
      │   │ PHYSICS ENGINE (Multi-threaded)         │
      │   │ ├─ Quaternion-based orientation        │
      │   │ ├─ Rigid body dynamics (inertia tensor)│
      │   │ ├─ apply_motor_torques()                │
      │   │ ├─ compute_reward_parallel() [NEW]     │
      │   │ ├─ Spring-damper contacts              │
      │   │ ├─ Contact-dependent gravity           │
      │   │ └─ Agent-centered world                │
      │   └──────────────────────────────────────────┘
      │
      ├─→ ┌──────────────────────────────────────────┐
      │   │ TRAINING ENGINE (PPO)                   │
      │   │ ├─ collect_trajectory()                 │
      │   │ ├─ train_on_trajectory()                │
      │   │ ├─ PPO loss computation                 │
      │   │ └─ World model training                 │
      │   └──────────────────────────────────────────┘
      │
      ├─→ ┌──────────────────────────────────────────┐
      │   │ RENDERER (PyRay)                        │
      │   │ ├─ 3D visualization                     │
      │   │ ├─ Quadruped rendering                 │
      │   │ ├─ Goal visualization                   │
      │   │ └─ Camera control                       │
      │   └──────────────────────────────────────────┘
      │
      └─→ ┌──────────────────────────────────────────┐
          │ NEURAL NETWORKS (entity.py)             │
          │ ├─ EntityBelief: Policy/Value heads    │
          │ │  ├─ Feature encoder (37D → 128D)    │
          │ │  ├─ LSTM (256D state)               │
          │ │  ├─ Policy head (256D → 12D)       │
          │ │  └─ Value head (256D → 1D)         │
          │ └─ WorldModel: Dynamics learning      │
          └──────────────────────────────────────────┘
```

### Training Loop (Unified)

```
START
  ↓
RESOURCE DETECTION
├─ Auto-detect NUM_ENVS from available memory
├─ Initialize multi-environment support (up to 64 envs)
├─ Setup physics thread pool (num_cpus - 1 workers)
└─ Setup device (GPU/CPU)
  ↓
FOR EACH EPISODE
  ├─ FOR EACH ENVIRONMENT (PARALLEL)
  │   ├─ Environment.reset() → neutral quadruped
  │   ├─ FOR EACH STEP (up to MAX_STEPS)
  │   │   ├─ obs = Environment.observe() [37D]
  │   │   ├─ action = Policy(obs) [12D motor torques] ← GPU
  │   │   ├─ reward = compute_reward_parallel() [← ThreadPool on CPU]
  │   │   │   ├─ Thread 1: apply_motor_torques + reward calc
  │   │   │   ├─ Thread 2: apply_motor_torques + reward calc
  │   │   │   └─ Thread N: apply_motor_torques + reward calc
  │   │   ├─ value = ValueNet(obs) ← GPU
  │   │   ├─ log_prob = Policy.log_prob(action, obs) ← GPU
  │   │   └─ Store (obs, action, reward, value, log_prob)
  │   └─ END STEP
  ├─ END ENVIRONMENTS (all done)
  │
  ├─ BATCH PROCESSING (GPU)
  │   ├─ Concatenate all trajectories
  │   ├─ Compute advantages (GAE across batch)
  │   └─ Normalize advantages
  │
  ├─ WORLD MODEL TRAINING (GPU with mixed precision)
  │   ├─ Learn dynamics: (obs, action) → next_obs
  │   ├─ Predict rewards
  │   └─ MSE loss on observations/rewards
  │
  ├─ PPO TRAINING (4 epochs)
  │   ├─ FOR EACH EPOCH
  │   │   ├─ Process each trajectory (LSTM state reset)
  │   │   ├─ Forward: policy(obs), value(obs)
  │   │   ├─ Compute policy loss (clipped PPO)
  │   │   ├─ Compute value loss (advantages)
  │   │   ├─ Add entropy bonus
  │   │   ├─ Total loss = policy + value - entropy
  │   │   └─ Backward + update
  │   └─ END EPOCH
  │
  ├─ CHECKPOINT & LOG
  │   ├─ Save model weights
  │   ├─ Log metrics (reward, loss, value)
  │   └─ Print progress
  │
  ├─ EVALUATION (every N episodes)
  │   ├─ Renderer.init()
  │   ├─ Run eval episodes with visualization
  │   └─ Renderer.close()
  │
└─ END EPISODE

END
```

## Component Details

### 1. System Class (simulate.py)
**Responsibility**: Main orchestrator, coordinates all components
└── Utility Methods
    ├── move(creature_id, dx, dy, dz)  # Velocity-based physics
    ├── observe(creature)
    ├── _compute_reward()
    ├── _distance_to_goal()
    ├── spawn_random_goal()
    ├── save_checkpoint()
    └── load_checkpoint()
```

### Performance: JIT Kernel Fusion Architecture

**The Critical Optimization Pattern**:
```
SLOW: Python → JIT_A() → Python → JIT_B() → Python → JIT_C() → Python
      [3 boundary crossings, context switches, overhead]

FAST: Python → [LARGE_JIT_KERNEL(physics + reward + obs)] → Python
      [1 boundary crossing, compiler optimizes everything together]
```

**Why Fusion Matters**:
- **Boundary Crossing Cost**: Each Python↔JIT transition has overhead (~1-2μs)
- **Scale**: In training, move() called ~millions of times (3+ boundary crossings each)
- **Solution**: Single large compiled kernel eliminates transition overhead
- **Location**: Kernel fused directly into simulate.py (zero import overhead)

**Kernel Specification** (simulation_step in simulate.py):
```python
@jit(nopython=True)
def simulation_step(
    # State (6 floats)
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    # Action (3 floats)
    accel_x, accel_y, accel_z,
    # World (9 floats)
    goal_x, goal_y, goal_z,
    bound_min_x, bound_min_y, bound_min_z,
    bound_max_x, bound_max_y, bound_max_z,
    # Physics params (11 floats)
    accel_scale_xy, accel_scale_z, max_accel, max_vel,
    momentum_damping, gravity, terminal_vel_z, ground_level,
    ground_friction, air_friction, air_drag,
    # Reward params (8 floats)
    prev_dist, goal_threshold, proximity_threshold,
    distance_reward_scale, proximity_bonus_scale, goal_bonus,
    wall_penalty_scale, stamina_penalty
) -> tuple:  # 15 floats: pos(3) + vel(3) + reward + dist + penalty + obs(5)
```

### Vectorization Details

**Auto-Resource Detection**:
```python
def detect_available_envs():
    gpu_mem_free = sum(gpu.memoryFree for gpu in GPUs)
    cpu_mem_free = psutil.virtual_memory().available
    total_free = gpu_mem_free + cpu_mem_free
    
    # Calculate how many envs fit in MAX_DATA_THRESHOLD_MB
    num_envs = min(max(total_free / mem_per_env, MIN_ENVS), MAX_ENVS)
    return num_envs
```

**Parallel Collection**:
- Each environment runs independently
- Trajectories collected simultaneously
- No blocking or synchronization needed
- Each env maintains separate RNN state

**Batch Training**:
- Concatenate trajectories from all envs
- Single backward pass updates weights
- Aggregated advantages & returns
- 3-5x fewer optimizer steps vs serial

### 2. EntityBelief Class (entity.py)
**Responsibility**: Policy and value function neural network

```
EntityBelief
├── Input: obs_dim=7 (relative pos, absolute pos, stamina)
│
├── Encoder
│   └── Linear(7) → ReLU → Linear(256) → ReLU → Linear(64)
│       Output: 64-dim feature vector
│
├── Graph Attention Network (GAT)
│   ├── 4 attention heads
│   ├── 32 dims per head
│   ├── Edge index: self-loop (single agent)
│   └── Output: 128-dim (4 heads × 32 dims, concatenated)
│
├── LSTM Layer
│   ├── Input: 128-dim
│   ├── Hidden size: 128
│   ├── Output (h, c): tuple of 128-dim tensors
│   └── Preserves state across episodes
│
├── Policy Head
│   ├── FC: 128 → 64 → 3
│   ├── Output: μ (mean action)
│   └── log_std: Global parameter (state-independent)
│
└── Value Head
    ├── FC: 128 → 64 → 1
    └── Output: V(s) scalar baseline
```

### 3. WorldModel Class (entity.py) - DreamerV3
**Responsibility**: Learn and predict environment dynamics

```
WorldModel
├── Input: obs (7D), action (3D continuous)
│
├── Encoder
│   └── Linear(7) → ReLU → Linear(256) → ReLU → Linear(128)
│       Output: 128-dim latent state
│
├── Dynamics Model
│   └── Linear(128+3) → ReLU → Linear(256) → ReLU → Linear(128)
│       Input: [latent_state, action]
│       Output: next_latent_state
│
├── Reward Predictor
│   └── Linear(128) → ReLU → Linear(256) → ReLU → Linear(1)
│       Output: predicted reward (scalar)
│
├── Done Predictor
│   └── Linear(128) → ReLU → Linear(256) → Linear(1) → Sigmoid
│       Output: done probability (0-1)
│
├── Decoder
│   └── Linear(128) → ReLU → Linear(256) → ReLU → Linear(7)
│       Output: reconstructed observation
│
└── Key Methods
    ├── encode(obs) → latent
    ├── decode(latent) → reconstructed_obs
    ├── predict_next(latent, action) → next_latent
    ├── predict_reward(latent) → reward
    ├── predict_done(latent) → done_prob
    └── forward(obs, action) → (next_latent, reward, done, next_obs_recon)
```

### 4. Creature Dataclass (entity.py)
**Responsibility**: Encapsulates single agent state

```
Creature
├── en_id: int                          # Entity ID (agent=1)
├── pos: torch.Tensor (3,)              # [x, y, z] position
├── orientation: torch.Tensor (3,)      # [pitch, yaw, roll]
├── rnn_state: (h, c) tuple             # LSTM hidden/cell states
└── stamina: torch.Tensor (scalar)      # Current energy level
```

## Data Flow

### Single Step Forward Pass

```
1. observe(creature)
   └─→ [rel_x, rel_y, rel_z, abs_x, abs_y, abs_z, norm_stamina]
       All normalized to [-1, 1] range

2. model(obs)  [EntityBelief forward]
   ├─→ Encoder: obs → 64-dim features
   ├─→ GAT: features → 128-dim with attention
   ├─→ LSTM: 128-dim → (h, c) with state preservation
   ├─→ Policy Head: (h, c) → μ (mean action 3D)
   └─→ Value Head: (h, c) → V(s) (scalar)

3. Sample action
   └─→ a = tanh(μ + σ * ε) where σ is global log_std

4. move(creature, action)
   ├─→ Calculate movement magnitude and type (walk vs jump)
   ├─→ Deduct stamina from creature
   ├─→ Clamp position to boundaries
   ├─→ Calculate wall penetration
   └─→ Return (wall_penalty, stamina_penalty)

5. _compute_reward(creature, prev_distance, wall_penalty, stamina_penalty, is_idle)
   ├─→ Current distance = distance_to_goal(creature)
   ├─→ Distance reward = (prev_distance - curr_distance) * scale
   ├─→ Goal bonus = 10.0 if within threshold
   ├─→ Proximity bonus = linear scale based on distance
   ├─→ Stamina regen bonus = 0.01 if idle
   └─→ total_reward = all components summed

6. Store transition
   └─→ trajectory.append({
       'obs': obs,
       'action': action,
       'reward': reward,
       'value': value.item(),
       'log_prob': log_prob
   })
```

### Trajectory Collection Loop

```
for step in range(max_steps):
    obs = observe(creature)                    # [1, 7]
    
    with torch.no_grad():
        mu, log_std, value = model(obs)
    
    # Sample action with exploration
    std = log_std.exp()
    eps = torch.randn_like(mu)
    action = torch.tanh(mu + std * eps)        # [-1, 1]
    log_prob = compute_log_prob(mu, log_std, action)
    
    # Execute action in environment
    wall_penalty, stamina_penalty = move(creature, action)
    reward, curr_distance = _compute_reward(...)
    
    # Check termination
    is_done = curr_distance < goal_threshold
    if is_done:
        spawn_random_goal()
    
    # Store for training
    trajectory.append({
        'obs': obs.cpu(),
        'action': action.cpu(),
        'reward': reward.cpu(),
        'value': value.cpu(),
        'log_prob': log_prob.cpu(),
        'done': torch.tensor(is_done)
    })
    
    prev_distance = curr_distance
```

### Training Loop

```
for episode in range(max_episodes):
    # Phase 1: Collect trajectory
    trajectory = collect_trajectory(max_steps=1000)
    
    # Phase 2: Train world model (optional)
    if world_model_enabled:
        wm_loss = train_world_model(trajectory)
    
    # Phase 3: Compute advantages
    values = [t['value'] for t in trajectory]
    rewards = [t['reward'] for t in trajectory]
    advantages = compute_gae(rewards, values, gamma=0.99, lambda=0.95)
    returns = advantages + values
    advantages = (advantages - mean) / std
    
    # Phase 4: PPO training epochs
    for epoch in range(num_epochs):
        for batch in minibatches(trajectory):
            # Forward pass
            new_mu, new_log_std, new_value = model(batch['obs'])
            new_action_dist = Normal(new_mu, new_log_std.exp())
            new_log_prob = new_action_dist.log_prob(batch['action']).sum(-1)
            
            # Importance sampling ratio
            ratio = (new_log_prob - batch['log_prob']).exp()
            
            # Clipped policy loss
            policy_loss = -min(
                ratio * batch['advantages'],
                clip(ratio, 1-eps, 1+eps) * batch['advantages']
            ).mean()
            
            # Value loss with clipping
            value_pred = new_value.squeeze()
            value_old = batch['value']
            value_clipped = value_old + clip(value_pred - value_old, -0.2, 0.2)
            value_loss = max(
                (value_pred - batch['returns'])**2,
                (value_clipped - batch['returns'])**2
            ).mean()
            
            # Entropy bonus
            entropy = -new_log_prob.mean()
            
            # Combined loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Phase 5: Save checkpoint
    save_checkpoint()
    
    # Phase 6: Logging
    episode_reward = sum(t['reward'] for t in trajectory)
    log_metrics(episode, episode_reward, losses)
```

## Memory Layout (Tensors)

### Creature State (GPU)
```
pos:            [1, 3]        float32  (x, y, z coordinates)
stamina:        [1]           float32  (0-200 energy)
rnn_state.h:    [1, 128]      float32  (LSTM hidden)
rnn_state.c:    [1, 128]      float32  (LSTM cell)
Total:          ~1.4 KB per creature
```

### Observation Batch (GPU)
```
obs:            [N, 7]        float32  (relative pos, absolute pos, stamina)
action:         [N, 3]        float32  (continuous 3D actions)
reward:         [N]           float32  (scalar rewards)
log_prob:       [N]           float32  (action log probabilities)
value:          [N]           float32  (baseline values)
done:           [N]           bool     (terminal flags)
Total:          ~320 bytes per timestep
```

### Model Parameters
```
EntityBelief:
  Encoder:      ~1.8M parameters
  GAT:          ~0.8M parameters
  LSTM:         ~2.1M parameters
  Policy Head:  ~0.3M parameters
  Value Head:   ~0.3M parameters
  Total:        ~5.3M parameters

WorldModel:
  Encoder:      ~0.8M parameters
  Dynamics:     ~0.8M parameters
  Decoders:     ~1.0M parameters
  Total:        ~2.6M parameters

Combined:       ~7.9M parameters (~32 MB on GPU)
```

## Performance Characteristics

### Time Complexity per Step
```
observe():           O(1)    ~0.1 ms
model.forward():     O(N)    ~1-2 ms (N=input dim)
move():              O(1)    ~0.1 ms
_compute_reward():   O(1)    ~0.2 ms
Total per step:             ~1.5 ms

Episode (1000 steps):       ~1.5 seconds
World model training:       ~0.5 seconds
PPO training:               ~0.3 seconds
Total per episode:          ~2.3 seconds
Throughput:                 ~430 episodes/hour
```

### Space Complexity
```
Trajectory buffer:    ~1000 * 320 bytes = 320 KB
Model parameters:     ~32 MB (on GPU)
Optimizer states:     ~64 MB (2x parameters for Adam)
Training data:        ~500 MB max (trajectory history)
Total GPU memory:     ~600 MB (relatively small)
```

## Optimization Decisions

### 1. Precomputed Constants
- Boundary tensors
- World scale factors
- Reward component weights
- Action scales
**Benefit**: Avoid repeated tensor creation in hot loops

### 2. Tensor Operations (No Float Conversions)
- All math stays in tensor domain
- GPU-native operations throughout
- Only convert to float for logging/comparison
**Benefit**: ~3-5x faster than repeated device transfers

### 3. GAT Single-Node Optimization
- Detects N==1 agent case
- Skips attention computation (returns identity)
**Benefit**: 50% speedup for single-agent scenarios

### 4. Global log_std Parameter
- Not state-dependent
- Single learnable parameter shared across policy
**Benefit**: Stability, prevents exploration collapse

### 5. Mixed Precision (on CUDA)
- Uses autocast for forward passes
- Full precision for backward
**Benefit**: ~2x memory savings, faster computation

## Extension Points

Want to extend the system? Key areas:

1. **New Reward Components**
   - Add to `_compute_reward()` method
   - Add config parameters to `Config` class

2. **New Observation Features**
   - Modify `observe()` method
   - Update obs_dim in System.__init__()
   - Update WorldModel enc/decoder dims

3. **New Neural Network Layers**
   - Modify EntityBelief class
   - Add to forward pass pipeline

4. **New Action Types**
   - Modify `move()` method
   - Update action space handling

See [ADVANCED_TOPICS.md](ADVANCED_TOPICS.md) for detailed extension guide.
