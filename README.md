# 3D Reinforcement Learning Navigation System

State-of-the-art 3D DRL agent with **quadruped morphology**, **advanced physics engine** (rigid body dynamics, quaternion orientation, constraint solvers), **PPO**, **LSTM**, **GAT**, **DreamerV3 world model**, and **vectorized multi-environment training**.

## Key Features

🚀 **Vectorized Multi-Env Training**: 4-8x faster data collection (auto-configured, respects system resources)
🌍 **Advanced Physics Engine**: Rigid body dynamics, quaternion rotations, sequential impulses, realistic contacts & friction
🦴 **Quadruped Robot**: 4 legs × 3 joints (12 DOF), forward kinematics, balance-focused control
🧠 **Advanced RL**: PPO + LSTM + GAT attention + DreamerV3 world model
⚡ **Agent-Centered World**: Agent always at origin, simplified learning and translation invariance

## Quick Start

```bash
python run.py       # Train the agent (auto-detects parallel envs)
python test.py      # Run tests
```

## System Overview

**Agent**: Navigates 3D world toward randomly spawned goals while managing stamina energy.

**Key Components**:
- **Agent System**: PPO policy + LSTM (128 hidden) + GAT (4 heads) + stamina tracking
- **World System**: 3D environment, 6-component reward, velocity-based physics
- **Physics**: Momentum, gravity, friction, air drag, collision detection
- **Training**: Auto-vectorized multi-environment (2-32 parallel envs based on available memory)
- **World Model**: DreamerV3 for learning dynamics predictions

## Architecture

| Component | Purpose | Status |
|-----------|---------|--------|
| PPO | Policy optimization algorithm | ✓ Active |
| LSTM (128 units) | Temporal reasoning | ✓ Active |
| GAT (4 heads) | Multi-head spatial attention | ✓ Active |
| WorldModel (DreamerV3) | Learn environment dynamics | ✓ Active |
| Vectorized Training | Parallel environment simulation | ✅ NEW |
| Velocity-Based Physics | Realistic movement with inertia | ✅ NEW |
| Stamina System | Energy resource management | ✓ Active |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Documentation

For detailed guides, see [docs/](docs/) folder:
- **[Quick Start](docs/QUICKSTART.md)**: Setup and first run
- **[Architecture](docs/ARCHITECTURE.md)**: System design, vectorization, physics
- **[Physics Engine](docs/PHYSICS_ENGINE_IMPROVED.md)**: Advanced rigid body dynamics, quaternions, constraint solvers
- **[Quadruped Task](docs/QUADRUPED_BALANCE_TASK.md)**: Robot morphology, forward kinematics, balance rewards
- **[Configuration](docs/CONFIGURATION.md)**: All tunable parameters + vectorization settings
- **[World System](docs/WORLD_SYSTEM.md)**: Environment, physics, rewards
- **[Agent System](docs/AGENT_SYSTEM.md)**: Agent design, PPO algorithm, learning
- **[World Model](docs/WORLD_MODEL.md)**: DreamerV3 implementation


## Running the System

```bash
python run.py       # Train the agent
python test.py      # Run tests
```

**Training**: Runs at unlimited speed (no rendering). Live episode counter and metrics logged to `training.log`.

**Evaluation**: After training, 5 evaluation episodes run with 3D rendering at 10 FPS.

## Camera Controls (Evaluation)

| Key | Action |
|-----|--------|
| **WASD/QE** | Move camera |
| **LMB + Drag** | Rotate camera |

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for camera control details and visual overlays.

## Configuration

Edit `Config` class in [simulate.py](simulate.py):

```python
DIM = (64, 64, 16)            # World dimensions
MAX_STEPS_PER_EPISODE = 1000  # Steps per episode
EVAL_EPISODES = 5             # Test episodes after training
ACTION_SCALE_Z = 0.2          # Vertical speed (horizontal is hardcoded to 1.0)
PPO_CLIP_RATIO = 0.2          # PPO clip range
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all parameters.

## Files

- [simulate.py](simulate.py) - Main training loop
- [entity.py](entity.py) - Neural network architecture
- [requirements.txt](requirements.txt) - Dependencies
- [training.log](training.log) - Training metrics (generated)

## Future Work

See [ROADMAP.md](ROADMAP.md) for planned improvements and research directions.

## Dependencies

```
numpy>=1.24.0
torch>=2.0.0
pyray>=0.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

## License

MIT