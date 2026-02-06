# Quickstart Guide - Train Quadruped Balance in 5 Minutes

## Installation (1 minute)

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (recommended, CPU also works but slower)

### Setup

```bash
# Navigate to project
cd reinforce

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m py_compile config.py source/entity.py source/simulate.py
```

## Run Training (2 minutes)

### Start Quadruped Balance Training

```bash
python run.py
```

**What happens:**
1. System initializes (creates 4-legged quadruped agent)
2. Training loop starts collecting trajectories (no rendering)
3. Real-time metrics display: episode number, reward, policy loss
4. Agent learns to balance and move toward goals
5. Training runs for MAX_TRAINING_EPISODES (default: 100 episodes)

### Monitor Real-Time Progress

In another terminal:
```bash
tail -f training.log
```

**Expected output:**
```
2026-02-04 10:30:00 - INFO - Episode 1: Collecting trajectory...
2026-02-04 10:30:01 - INFO - Ep 1: reward=0.23, policy_loss=0.456, value_loss=0.123
2026-02-04 10:30:02 - INFO - Episode 2: Collecting trajectory...
...
```

## What to Expect During Training

### Episodes 1-10
- Rewards: 0-5 (agent learning basic control)
- Agent: Jerky movements, falls frequently
- Loss: High, decreasing

### Episodes 20-50
- Rewards: 10-30 (agent learning balance)
- Agent: More stable stance, starts moving toward goals
- Loss: Decreasing smoothly

### Episodes 50+
- Rewards: 30-80+ (agent optimizing balance and movement)
- Agent: Smooth walking, stable balance
- Loss: Stable and low

### Evaluation Phase

After MAX_TRAINING_EPISODES:
1. 3D visualization opens
2. Watch quadruped balance and walk toward goals
3. System runs EVAL_EPISODES (default: 2)
4. Window closes after evaluation

## Key Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| reward | Total episode reward | +20 to +100 |
| policy_loss | Policy gradient loss | 0.01-0.1 |
| value_loss | Value estimation error | 0.01-0.1 |
| actor_loss | Actor network loss | < 0.1 |

## Configuration Tuning (Optional)

Edit `config.py` for common adjustments:

```python
# Training duration
MAX_TRAINING_EPISODES = 200    # More training: default is 100
MAX_STEPS_PER_EPISODE = 500    # Longer episodes: default is 500

# Physics tuning
GRAVITY = 9.8                  # Increase for harder balance
CONTACT_STIFFNESS = 0.5        # Increase for stiffer ground
CONTACT_DAMPING = 0.2          # Increase for less bouncy

# Reward tuning
COM_REWARD_SCALE = 10.0        # Increase to prioritize goal reaching
CONTACT_REWARD = 0.1           # Increase for more stable stance
ENERGY_PENALTY = 0.01          # Increase to discourage jerky movements
```

Then run `python run.py` again.

## Common Adjustments

### "I want faster training"
```python
MAX_TRAINING_EPISODES = 50      # Fewer episodes, check convergence sooner
```

### "I want better balance control"
```python
CONTACT_REWARD = 0.5           # Reward for foot contact (default: 0.1)
COM_REWARD_SCALE = 5.0         # Reduce goal emphasis, focus on balance
```

### "I want agent to reach goals faster"
```python
COM_REWARD_SCALE = 20.0        # Increase goal reward (default: 10.0)
CONTACT_REWARD = 0.05          # Reduce stability emphasis
```

### "I want smoother movements"
```python
ENERGY_PENALTY = 0.05          # Increase motor cost (default: 0.01)
JOINT_DAMPING = 0.02           # Increase joint friction (default: 0.01)
```

### "Agent keeps falling"
```python
CONTACT_STIFFNESS = 1.0        # Increase ground stiffness (default: 0.5)
CONTACT_DAMPING = 0.5          # Increase ground damping (default: 0.2)
```

## Next Steps

1. **See physics details**: [PHYSICS_AND_WORLD.md](PHYSICS_AND_WORLD.md)
2. **Understand quadruped design**: [QUADRUPED_BALANCE_TASK.md](QUADRUPED_BALANCE_TASK.md)
3. **Learn agent architecture**: [AGENT_SYSTEM.md](AGENT_SYSTEM.md)
4. **Explore all config options**: [CONFIGURATION.md](CONFIGURATION.md)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pyray'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
```python
# In config.py, reduce parallel environments
MAX_ENVS = 4  # Default is 16
```

### "Training hangs or crashes"
1. Check `training.log` for error messages
2. Try `Ctrl+C` and restart
3. Ensure `python -m py_compile source/simulate.py` works first

### "Agent doesn't improve"
1. Check that reward values are reasonable (10-50 range)
2. Try increasing MAX_TRAINING_EPISODES to 200+
3. Review [CONFIGURATION.md](CONFIGURATION.md) reward tuning section
4. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## Full Documentation Index

- 📖 [Complete Index](INDEX.md)
- 🏗️ [Architecture](ARCHITECTURE.md)
- ⚡ [Stamina System](STAMINA_SYSTEM.md)
- 🌐 [World Model](WORLD_MODEL.md)
- 📊 [Training Guide](TRAINING_GUIDE.md)
- ⚙️ [Configuration](CONFIGURATION.md)
- 🔧 [API Reference](API_REFERENCE.md)
- 🐛 [Troubleshooting](TROUBLESHOOTING.md)

**Happy training! 🚀**
