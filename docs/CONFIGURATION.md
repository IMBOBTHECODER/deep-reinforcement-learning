# Configuration Reference - Quadruped Balance Task

Complete documentation of all configuration options. Default values are tuned and work well.

## Quick Start

For most users: **Use defaults**, adjust only these:
- **Longer training**: `MAX_TRAINING_EPISODES = 200`
- **Better balance**: `CONTACT_REWARD = 0.2`
- **Reach goals faster**: `COM_REWARD_SCALE = 20.0`
- **Smoother movements**: `ENERGY_PENALTY = 0.05`

## Training Control

```python
MAX_TRAINING_EPISODES = 100
# Episodes before evaluation
# Range: 10-500

MAX_STEPS_PER_EPISODE = 500
# Steps per episode
# Range: 100-2000

LOAD_CHECKPOINT = True
# Resume training from checkpoint
```

## Environment

```python
# Vectorized parallel training
USE_VECTORIZED_ENV = True
NUM_ENVS = None           # Auto-detect (2-16)
MAX_ENVS = 16
MIN_ENVS = 2

# World layout
DIM = (64, 32, 64)        # Width × Height × Depth

# Visualization
SCALE = 32                # Pixels per unit (8-64)
TARGET_FPS = 60
EVAL_EPISODES = 2
RUN_EVALUATION = True
EVAL_STEPS_PER_SEC = 20
```

## Neural Network

```python
OBS_DIM = 37              # Fixed: 12 angles + 12 velocities + 4 contacts + 3 orient + 3 COM + 3 goal-rel
ACTION_DIM = 12           # Fixed: 12 motor torques (3 per leg × 4 legs)

EMBED_DIM = 128           # Feature encoder output (64-256)
LSTM_HIDDEN = 256         # LSTM state size (128-512)
```

## PPO Hyperparameters

```python
BATCH_SIZE = 64           # Batch size (16-256)
PPO_EPOCHS = 4            # Training passes (1-10)
PPO_CLIP_RATIO = 0.2      # Clipping threshold (0.1-0.5)

GAMMA = 0.99              # Discount factor (0.95-0.999)
GAE_LAMBDA = 0.95         # Advantage estimation (0.9-0.99)
ENTROPY_COEF = 0.01       # Exploration bonus (0.001-0.1)
VALUE_COEF = 0.5          # Value loss weight (0.1-1.0)

LR = 3e-4                 # Learning rate (1e-5 to 1e-3)
```

## Physics Parameters (Critical)

### Joint Dynamics
```python
JOINT_DAMPING = 0.01
# Angular velocity damping
# Higher = smoother, Lower = oscillatory

MAX_TORQUE = 5.0
# Maximum motor torque (N·m)
# Higher = stronger motors, easier balance

SEGMENT_LENGTH = 0.1
# Leg segment length (m)
# Total leg = 3 * 0.1 = 0.3m
```

### Gravity & Ground
```python
GRAVITY = 9.8
# Gravitational acceleration (m/s²)
# Higher = harder balance, Lower = easier

GROUND_FRICTION_COEFFICIENT = 0.8
# Friction (prevents sliding)

GROUND_LEVEL = 0.0
# Ground plane Z-coordinate
```

### Spring-Damper Contact Model
```python
CONTACT_STIFFNESS = 0.5
# Ground spring stiffness
# Higher = stiffer, Lower = softer
# Tune if: feet sink (increase) or too bouncy (decrease)

CONTACT_DAMPING = 0.2
# Energy dissipation on impact
# Higher = absorbs more, Lower = bouncy

FOOT_HEIGHT_THRESHOLD = 0.05
# Distance to detect ground contact (m)
```

## Reward Function

```python
COM_DISTANCE_THRESHOLD = 0.3
# Distance threshold for goal closeness (m)

COM_REWARD_SCALE = 10.0
# Goal reaching reward
# Higher = prioritize reaching, Lower = prioritize balance

CONTACT_REWARD = 0.1
# Reward per foot in ground contact
# Higher = stable stance, Lower = less stability emphasis

ENERGY_PENALTY = 0.01
# Penalty per unit motor torque
# Higher = efficient movements, Lower = allow jerky control

# Legacy (kept for compatibility)
STABILITY_REWARD_SCALE = 1.0
GOAL_BONUS = 10.0
```

## Common Adjustments

### Faster Training
```python
MAX_TRAINING_EPISODES = 50
RUN_EVALUATION = False
PPO_EPOCHS = 2
```

### Better Balance
```python
CONTACT_REWARD = 0.3           # Increase stability
COM_REWARD_SCALE = 5.0         # Reduce goal emphasis
ENERGY_PENALTY = 0.05          # Penalize jerky movements
```

### Reach Goals Faster
```python
COM_REWARD_SCALE = 25.0        # High goal reward
CONTACT_REWARD = 0.05          # Low stability reward
```

### Agent Falls Too Much
```python
CONTACT_STIFFNESS = 1.0        # Stiffer ground
CONTACT_DAMPING = 0.3          # More damping
GRAVITY = 5.0                  # Lower gravity
```

### Smoother Movements
```python
ENERGY_PENALTY = 0.05          # Higher penalty
JOINT_DAMPING = 0.02           # More friction
```

### Limited GPU Memory
```python
MAX_ENVS = 4
BATCH_SIZE = 32
LSTM_HIDDEN = 128
```

### Agent Gets Stuck
```python
ENTROPY_COEF = 0.05            # More exploration
LR = 5e-4                      # Higher learning rate
```

## World Model (Optional)

```python
WORLD_MODEL_LATENT_DIM = 128
# Latent space dimension (64-256)

WORLD_MODEL_HIDDEN_DIM = 256
# Hidden layer size (128-512)

WORLD_MODEL_LR = 1e-3
# Learning rate (1e-4 to 1e-2)

WORLD_MODEL_WEIGHT_DECAY = 1e-6
# L2 regularization

IMAGINATION_HORIZON = 15
# Planning rollout length (5-50)
```

## Quick Reference Table

| Parameter | Default | Range | Adjust For |
|-----------|---------|-------|---|
| MAX_TRAINING_EPISODES | 100 | 10-500 | Training duration |
| MAX_STEPS_PER_EPISODE | 500 | 100-2000 | Episode length |
| NUM_ENVS | auto | 2-32 | Speed/memory trade-off |
| GRAVITY | 9.8 | 1-20 | Balance difficulty |
| CONTACT_STIFFNESS | 0.5 | 0.1-2.0 | Ground hardness |
| COM_REWARD_SCALE | 10.0 | 1-50 | Goal vs balance |
| CONTACT_REWARD | 0.1 | 0.01-1.0 | Stability emphasis |
| ENERGY_PENALTY | 0.01 | 0.001-0.1 | Movement smoothness |
| LR | 3e-4 | 1e-5-1e-3 | Learning speed |
| ENTROPY_COEF | 0.01 | 0.001-0.1 | Exploration level |

## Tips

1. **Start with defaults** - They're already tuned
2. **Change one parameter at a time** - Know what actually helps
3. **Monitor training.log** - Check loss stability
4. **Be patient** - Changes take 50+ episodes to show effect
5. **Document changes** - Keep notes of what worked

## Related Documentation

- **Physics details**: [PHYSICS_AND_WORLD.md](PHYSICS_AND_WORLD.md)
- **Quadruped design**: [QUADRUPED_BALANCE_TASK.md](QUADRUPED_BALANCE_TASK.md)
- **Agent architecture**: [AGENT_SYSTEM.md](AGENT_SYSTEM.md)
- **Quick start**: [QUICKSTART.md](QUICKSTART.md)
