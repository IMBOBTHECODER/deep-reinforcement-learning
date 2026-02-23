# Pattern A: On-Policy PPO + Off-Policy DreamerV3

## Problem Solved

**Previous Architecture (BROKEN)**:
- PPO trained on fresh rollouts → wants to discard old experiences
- DreamerV3 trained on same experiences → wants to keep replay buffer
- Single encoder updating from both on-policy and off-policy losses
- Result: **Representation drift, unstable training**

## Pattern A Solution

**Separate objectives with shared representation**:

### 1. Shared Encoder (Trainable ONLY by Dreamer)
```
Encoder (34D obs → 128D latent)
  ↓ (frozen path)
PPO Policy (uses frozen latent)
  ↓ (trainable path)
  updated by train_world_model() only
DreamerV3 World Model
```

### 2. Training Flow

**During each training step**:

```python
# Step 1: Collect trajectories with FROZEN encoder
rollouts = collect_trajectories_vectorized()

# Step 2: Train Dreamer (UNFREEZE encoder)
unfreeze_encoder()
for traj in rollouts:
    # Trains: encoder + dynamics + reward_head + done_head
    train_world_model(traj)  
freeze_encoder()

# Step 3: Train PPO (KEEP encoder FROZEN)
for traj in rollouts:
    # Trains: policy_mu + log_std + value_head
    # Encoder is `with torch.no_grad()` in forward pass
    train_ppo_policy(traj)
```

### 3. Key Implementation Details

#### Encoder Freezing
- **Created**: `freeze_encoder()` / `unfreeze_encoder()` methods
- **Default**: Encoder starts FROZEN (PPO can't update it)
- **During Dreamer training**: Temporarily unfrozen, then re-frozen

#### Separate Optimizers
```python
# WorldModel optimizer: trains ENCODER + dynamics + heads
world_model_optimizer = Adam(
    encoder_params + world_model_only_params,
    lr=WORLD_MODEL_LR
)

# PPO optimizer: trains ONLY policy (frozen encoder)
ppo_optimizer = Adam(
    policy_params,  # EXCLUDES encoder
    lr=PPO_LR
)
```

#### Encoder Forward Pass in PPO
```python
def forward(self, obs):
    # In PPOPolicy.forward():
    with torch.no_grad():  # Frozen
        latent = self.encoder(obs)
    
    # ... rest of PPO (can update)
    lstm_out = self.lstm(latent)
    mu = self.policy_mu(lstm_out)  # Updates this
    v = self.value(lstm_out)        # Updates this
```

### 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  Observation (34D)                                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Shared Encoder     │  <- Trained by Dreamer ONLY
        │  (128D latent)      │  <- Frozen during PPO training
        └─────────┬───────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
   ┌─────────┐          ┌────────────┐
   │ PPO     │          │ DreamerV3  │
   │ Policy  │          │ World Model│
   │ (trains)│          │ (trains    │
   │         │          │  encoder)  │
   └─────────┘          └────────────┘
     LSTM                Dynamics +
     + policy_mu         Reward Head
     + value             + Done Head
```

### 5. When Each Component Updates

| Component | Updates During | Rationale |
|-----------|---|---|
| Encoder | `train_world_model()` | Only Dreamer (off-policy expert) updates representations |
| LSTM | `train_ppo_policy()` | PPO operates on frozen latent |
| policy_mu | `train_ppo_policy()` | PPO controls via frozen encoder |
| log_std | `train_ppo_policy()` | PPO exploration strategy |
| value | `train_ppo_policy()` | PPO baseline |
| Dynamics | `train_world_model()` | Learn next latent prediction |
| Reward head | `train_world_model()` | Learn reward prediction |
| Done head | `train_world_model()` | Learn termination |
| Decoder | `train_world_model()` | Learn reconstruction |

## Migration Notes

### Was:
```python
self.model = EntityBelief(...)  # encoder + policy all mixed
self.world_model = WorldModel(...)  # separate encoder inside
```

### Now:
```python
self.model = EntityBelief(...)  # has model.encoder + model.policy
self.world_model = WorldModel(..., encoder=self.model.encoder)  # SHARED
```

### Code Changes
1. **EntityBelief.__init__()**: Split into `encoder + policy`, freeze encoder by default
2. **WorldModel.__init__()**: Accept optional `encoder` parameter (defaults to create own)
3. **TrainingEngine.__init__()**: Pass shared encoder to WorldModel
4. **train_on_trajectory()**: Unfreeze → Dreamer → Freeze → PPO pattern

## Expected Improvements

✅ **Representation stability**: Encoder trained by single objective (Dreamer on offline data)  
✅ **PPO stability**: Uses clean, frozen latent features without drift  
✅ **Off-policy advantage**: Dreamer trains on full replay buffer (future work: implement true replay)  
✅ **Tuning simplicity**: No more "why is representation changing unpredictably?"

## Next Steps (True Off-Policy)

Current: Dreamer trains on same on-policy rollouts as PPO  
Future: Implement actual replay buffer → Dreamer trains on past experiences → PPO on fresh data
