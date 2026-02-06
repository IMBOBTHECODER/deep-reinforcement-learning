# World Model (DreamerV3) - Comprehensive Guide

Deep dive into the DreamerV3 world model learning system that enables planning and faster training.

## Why World Models?

### Problem with Standard RL
```
Standard PPO:
┌──────────────────────────────────┐
│  Agent makes action               │
└──────────────────────────────────┘
           ↓ (must wait)
┌──────────────────────────────────┐
│  Environment responds              │
│  (real interaction needed)        │
└──────────────────────────────────┘
           ↓ (slow, expensive)
Learns policy from real rollouts

Problem: Every action requires environment interaction
- Slow training
- Sample inefficient
- Many wasted actions
- Real robots: expensive!
```

### Solution: World Model Learning
```
With World Model:
┌──────────────────────────────────┐
│  Agent makes action               │
└──────────────────────────────────┘
           ↓ (instant prediction)
┌──────────────────────────────────┐
│  World Model predicts next state  │
│  (no env needed, learned!)        │
└──────────────────────────────────┘
           ↓ (fast, free)
Multiple rollouts in imagination
- 10-100x faster iteration
- Sample efficient
- Discovers better strategies
- Better for real robots
```

## DreamerV3 Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│              World Model (DreamerV3)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Encoder: Observation → Latent State                │
│  ┌─────────────────────────────────────┐            │
│  │ Linear(7) → ReLU → Linear(256)      │            │
│  │ → ReLU → Linear(128)                │            │
│  │ Output: z_t (128-dim latent)        │            │
│  └─────────────────────────────────────┘            │
│           ↓                                          │
│  Dynamics: (z_t, a_t) → z_{t+1}                    │
│  ┌─────────────────────────────────────┐            │
│  │ Linear(128+3) → ReLU → Linear(256)  │            │
│  │ → ReLU → Linear(128)                │            │
│  │ Input: [latent_state, action]       │            │
│  │ Output: next_latent_state           │            │
│  └─────────────────────────────────────┘            │
│           ↓↓↓ (three prediction heads)              │
│           ├─→ Reward Head (latent → reward)        │
│           ├─→ Done Head (latent → terminal prob)   │
│           └─→ Decoder (latent → obs)               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Detailed Components

#### 1. Encoder
**Purpose**: Map high-dimensional observation to compact latent space

```python
class Encoder(nn.Module):
    def __init__(self, obs_dim=7, latent_dim=128):
        self.net = Sequential(
            Linear(obs_dim, 256),    # 7 → 256
            ReLU(),
            Linear(256, 256),        # 256 → 256
            ReLU(),
            Linear(256, latent_dim)  # 256 → 128
        )
    
    def forward(self, obs):
        # obs: [batch, 7]
        # output: [batch, 128]
        return self.net(obs)
```

**Why Latent Space?**
- Compression: 7D → 128D (actually expansion, but in meaningful dimensions)
- Abstraction: Learns compressed representation
- Efficiency: Faster dynamics computation
- Transfer: Can share across different tasks

#### 2. Dynamics Model
**Purpose**: Learn how latent state evolves with actions

```python
class Dynamics(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3):
        self.net = Sequential(
            Linear(latent_dim + action_dim, 256),    # 128+3=131 → 256
            ReLU(),
            Linear(256, 256),                         # 256 → 256
            ReLU(),
            Linear(256, latent_dim)                   # 256 → 128
        )
    
    def forward(self, latent, action):
        # latent: [batch, 128]
        # action: [batch, 3]
        combined = cat([latent, action], dim=-1)     # [batch, 131]
        next_latent = self.net(combined)              # [batch, 128]
        return next_latent
```

**Training Target**: 
- Learn to predict next state in latent space
- Supervised learning with encoder as target
- Loss: MSE(predicted_latent, target_latent)

#### 3. Reward Predictor
**Purpose**: Learn reward structure from latent states

```python
class RewardHead(nn.Module):
    def __init__(self, latent_dim=128):
        self.net = Sequential(
            Linear(latent_dim, 256),
            ReLU(),
            Linear(256, 1)
        )
    
    def forward(self, latent):
        # latent: [batch, 128]
        # output: [batch, 1] (scalar reward)
        return self.net(latent)
```

**Training Target**:
- Predict actual reward received
- Supervised learning from trajectory
- Loss: MSE(predicted_reward, actual_reward)

#### 4. Done Predictor
**Purpose**: Learn episode termination probability

```python
class DoneHead(nn.Module):
    def __init__(self, latent_dim=128):
        self.net = Sequential(
            Linear(latent_dim, 256),
            ReLU(),
            Linear(256, 1)
        )
    
    def forward(self, latent):
        # latent: [batch, 128]
        # output: [batch, 1] (probability 0-1)
        return Sigmoid(self.net(latent))
```

**Training Target**:
- Predict if episode will end
- Supervised learning from done flags
- Loss: BCE(predicted_done_prob, actual_done)

#### 5. Decoder
**Purpose**: Reconstruct observation from latent (reconstruction loss)

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, obs_dim=7):
        self.net = Sequential(
            Linear(latent_dim, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, obs_dim)
        )
    
    def forward(self, latent):
        # latent: [batch, 128]
        # output: [batch, 7] (reconstructed obs)
        return self.net(latent)
```

**Training Target**:
- Reconstruct original observation
- Acts as regularizer (forces meaningful latents)
- Loss: MSE(reconstructed_obs, original_obs)

## Training Pipeline

### Phase 1: Collect Trajectory (Real Environment)

```python
def collect_trajectory(max_steps=1000):
    trajectory = []
    creature.stamina = max_stamina
    
    for step in range(max_steps):
        obs = observe(creature)              # Real observation
        
        # Policy inference (no gradient)
        with no_grad():
            action = policy(obs)
        
        # Real environment step
        reward = move(creature, action)      # Real reward
        done = check_goal_reached()          # Real termination
        
        # Store transition
        trajectory.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done
        })
        
        if done:
            break
    
    return trajectory
```

### Phase 2: Train World Model

```python
def train_world_model(trajectory, epochs=1):
    optimizer = Adam(world_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # Sample transitions from trajectory
        for batch in minibatches(trajectory):
            obs_t = batch['obs']              # [B, 7]
            action_t = batch['action']        # [B, 3]
            reward_t = batch['reward']        # [B]
            obs_next = batch['next_obs']      # [B, 7]
            
            # Encode current observation
            latent_t = world_model.encode(obs_t)  # [B, 128]
            
            # Predict next state
            latent_pred = world_model.predict_next(latent_t, action_t)
            
            # Decode to reconstruct
            obs_recon = world_model.decode(latent_t)
            obs_next_recon = world_model.decode(latent_pred)
            
            # Predict reward and done
            reward_pred = world_model.predict_reward(latent_pred)
            done_pred = world_model.predict_done(latent_pred)
            
            # Compute losses
            recon_loss = MSE(obs_recon, obs_t)
            
            next_latent_target = world_model.encode(obs_next).detach()
            dynamics_loss = MSE(latent_pred, next_latent_target)
            
            reward_loss = MSE(reward_pred, reward_t)
            
            # Combined loss (weighted)
            total_loss = (
                1.0 * recon_loss +
                1.0 * dynamics_loss +
                1.0 * reward_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()
    
    return total_loss.item()
```

### Phase 3: Imagination Rollouts (Optional)

```python
def imagine_trajectory(obs, horizon=15):
    """Use world model to predict future without env interaction"""
    
    world_model.eval()
    with no_grad():
        latent = world_model.encode(obs)
        imagined_rewards = []
        
        for step in range(horizon):
            # Sample random action for exploration
            action = sample_random_action()
            
            # Predict next state
            latent = world_model.predict_next(latent, action)
            
            # Predict reward
            reward = world_model.predict_reward(latent)
            imagined_rewards.append(reward)
    
    return stack(imagined_rewards)  # [horizon]
```

**Use Cases:**
- Evaluate long-horizon consequences
- Bootstrap value function
- Plan ahead using MCTS
- Generate synthetic training data

## Training Details

### Batch Construction

```python
def create_training_batch(trajectory):
    """Convert trajectory to training batch"""
    
    # Pair transitions: (t, t+1)
    batch = {
        'obs': stack([t['obs'] for t in trajectory[:-1]]),      # [N, 7]
        'action': stack([t['action'] for t in trajectory[:-1]]), # [N, 3]
        'reward': stack([t['reward'] for t in trajectory[:-1]]), # [N]
        'next_obs': stack([t['obs'] for t in trajectory[1:]]),   # [N, 7]
        'done': stack([t['done'] for t in trajectory[:-1]])      # [N]
    }
    
    return batch
```

### Loss Functions

#### Reconstruction Loss
```python
recon_loss = MSE(obs_reconstructed, obs_original)
# Measures how well decoder preserves information
# High loss → encoder losing information
```

#### Dynamics Loss
```python
next_latent_target = encode(next_obs).detach()
dynamics_loss = MSE(predicted_next_latent, next_latent_target)
# Measures how well dynamics model predicts next state
# High loss → poor forward model
```

#### Reward Loss
```python
reward_loss = MSE(predicted_reward, actual_reward)
# Measures reward prediction accuracy
# High loss → learning bad reward structure
```

### Hyperparameters

```python
WORLD_MODEL_LR = 1e-3          # Learning rate (higher than policy 3e-4)
WORLD_MODEL_WEIGHT_DECAY = 1e-6  # L2 regularization
IMAGINATION_HORIZON = 15       # Steps to imagine ahead
WORLD_MODEL_LOSS_SCALE = 1.0   # Weight in combined loss
```

**Tuning:**
- Higher LR → Faster learning but less stable
- Lower LR → Slower learning but more stable
- Higher weight_decay → More regularization, smoother predictions
- Longer horizon → More aggressive planning, slower training

## Evaluation

### Prediction Quality

```python
# After training, test on held-out transitions
test_batch = create_batch(test_trajectory)

with no_grad():
    # Encode
    latent_t = world_model.encode(test_batch['obs'])
    
    # Predict next
    latent_pred = world_model.predict_next(latent_t, test_batch['action'])
    latent_target = world_model.encode(test_batch['next_obs'])
    
    # Compute error
    prediction_error = MSE(latent_pred, latent_target).item()
    print(f"Prediction MSE: {prediction_error:.4f}")
```

### Reward Accuracy

```python
reward_pred = world_model.predict_reward(latent_pred)
reward_mae = MAE(reward_pred, test_batch['reward']).item()
print(f"Reward MAE: {reward_mae:.4f}")
```

### Reconstruction Quality

```python
obs_recon = world_model.decode(latent_t)
recon_error = MSE(obs_recon, test_batch['obs']).item()
print(f"Reconstruction MSE: {recon_error:.4f}")
```

## Common Issues

### Issue: Prediction Accuracy Plateaus
**Cause**: Model capacity insufficient or training not converging
**Solution**:
- Increase hidden layer sizes (256 → 512)
- Train longer (more epochs)
- Increase learning rate slightly
- Check for NaN/Inf in losses

### Issue: Reconstruction Terrible
**Cause**: Encoder losing information or decoder underpowered
**Solution**:
- Increase latent dimension (128 → 256)
- Add residual connections
- Increase decoder capacity
- Check for gradient vanishing

### Issue: Reward Predictions Always Same
**Cause**: Reward head learning mean instead of variance
**Solution**:
- Initialize reward head to predict mean + small noise
- Add separate uncertainty output
- Use better reward normalization
- Increase reward head depth

### Issue: World Model Not Used
**Cause**: Policy training much faster, imagination unused
**Solution**:
- Reduce policy learning rate
- Increase world model training frequency
- Use world model for value bootstrapping
- Implement world model-based planning

## Advanced Techniques

### Multi-step Prediction
```python
# Predict multiple steps ahead
latent = encode(obs)
trajectory = [latent]

for _ in range(horizon):
    latent = dynamics(latent, sample_action())
    trajectory.append(latent)

# Use for long-term planning
```

### Uncertainty Estimation
```python
# Predict both mean and variance
mean, variance = world_model(obs, action)

# Use uncertainty for exploration
# High uncertainty → explore more
```

### Model Ensemble
```python
# Train multiple independent models
models = [WorldModel() for _ in range(5)]

# Ensemble predictions
predictions = [m.predict_next(latent, action) for m in models]
mean_pred = mean(predictions)
uncertainty = var(predictions)
```

## Debugging World Model

### Log Losses During Training

```python
if world_model_enabled:
    recon_loss = ...
    dynamics_loss = ...
    reward_loss = ...
    
    logger.info(f"WM: recon={recon_loss:.4f}, "
                f"dynamics={dynamics_loss:.4f}, "
                f"reward={reward_loss:.4f}")
```

### Visualize Predictions

```python
import matplotlib.pyplot as plt

# Compare predicted vs actual
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Position prediction
axes[0].scatter(actual_pos, predicted_pos)
axes[0].set_title('Position Prediction')

# Reward prediction
axes[1].scatter(actual_reward, predicted_reward)
axes[1].set_title('Reward Prediction')

# Reconstruction
axes[2].scatter(original_obs, reconstructed_obs)
axes[2].set_title('Observation Reconstruction')

plt.show()
```

## Future Enhancements

- [ ] Use imagination for Monte Carlo Tree Search (MCTS) planning
- [ ] Value bootstrapping from world model
- [ ] Model uncertainty for exploration
- [ ] Dynamics consistency regularization
- [ ] Separate model for each action dimension
- [ ] Attention mechanism over imagination

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - PPO training details
- Original Paper: Dreamer: Scalable Belief Exploration with World Models
- DreamerV3: Mastering Diverse Domains through World Models
