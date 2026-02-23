# Quadruped Balance System - Documentation Index

Welcome! This folder contains comprehensive documentation for the agent-centered quadruped balance task with advanced physics simulation.

## 📚 Documentation Structure

### Getting Started (Quick Links)
- **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in 5 minutes
- **[CONFIGURATION.md](CONFIGURATION.md)** - All configuration parameters explained
- **[PHYSICS.md](PHYSICS.md)** - Comprehensive physics guide (13 sections, reference doc)

### Latest Feature Updates (February 2026)
- **[PHYSICS_ENGINE_UPGRADES.md](PHYSICS_ENGINE_UPGRADES.md)** - 250 Hz, restitution, friction cones, vectorization
- **[JOINT_CONSTRAINTS.md](JOINT_CONSTRAINTS.md)** - Realistic joint types (revolute, spherical, fixed, hinge2, prismatic) with integrated examples
- **[ISAAC_GYM_COMPARISON.md](ISAAC_GYM_COMPARISON.md)** - Comparison to NVIDIA Isaac Gym standard
- **[ISAACLAB_COMPARABILITY.md](ISAACLAB_COMPARABILITY.md)** - IsaacLab v1.0 compatibility analysis

### Core Systems & Architecture
- **[QUADRUPED_BALANCE_TASK.md](QUADRUPED_BALANCE_TASK.md)** - Quadruped design, 12 DOF forward kinematics, balance-focused rewards
- **[AGENT_SYSTEM.md](AGENT_SYSTEM.md)** - Agent neural networks, policy/value heads, motor control
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, 5-class modular architecture, data flow
- **[DRL_PHYSICS_PLUGIN.md](DRL_PHYSICS_PLUGIN.md)** - How DRL integrates with physics engine

### Physics Subsystems (Specialized Topics)
- **[PHYSICS_CONFIG_GUIDE.md](PHYSICS_CONFIG_GUIDE.md)** - Physics parameter tuning guide
- **[ACTUATOR_LAG.md](ACTUATOR_LAG.md)** - Motor response time, lag dynamics
- **[FRICTION_MODELS.md](FRICTION_MODELS.md)** - Friction models and comparison
- **[ENERGY_TRACKING.md](ENERGY_TRACKING.md)** - Power consumption monitoring

### Training & AI
- **[WORLD_MODEL.md](WORLD_MODEL.md)** - DreamerV3 world model learning
- **[EVALUATION.md](EVALUATION.md)** - Evaluation metrics and procedures

### Performance & Design
- **[PERFORMANCE.md](PERFORMANCE.md)** - Optimization, profiling, benchmarks

## 🎯 Quick Links by Use Case

### "I want to start training NOW"
→ [QUICKSTART.md](QUICKSTART.md) (2 min)

### "I want realistic joint constraints"
→ [JOINT_CONSTRAINTS.md](JOINT_CONSTRAINTS.md) - See "Examples & Usage" section (5 min, 4 lines of code!)

### "I want to understand the physics system"
→ [PHYSICS.md](PHYSICS.md) (comprehensive 13-part guide)

### "I want to compare with industry standards"
→ [ISAAC_GYM_COMPARISON.md](ISAAC_GYM_COMPARISON.md) (NVIDIA Isaac Gym)  
→ [ISAACLAB_COMPARABILITY.md](ISAACLAB_COMPARABILITY.md) (NVIDIA IsaacLab)

### "I want to tune physics for a specific use case"
→ [PHYSICS_CONFIG_GUIDE.md](PHYSICS_CONFIG_GUIDE.md) (parameter tuning)  
→ [ACTUATOR_LAG.md](ACTUATOR_LAG.md) (motor response)  
→ [FRICTION_MODELS.md](FRICTION_MODELS.md) (ground friction)  
→ [ENERGY_TRACKING.md](ENERGY_TRACKING.md) (power consumption)

## 📋 File Map

## 📋 File Map (19 Total Documents)

| Document | Focus | Use For |
|----------|-------|---------|
| **QUICKSTART.md** | Setup & first training | Getting started quickly |
| **CONFIGURATION.md** | All parameters | Tuning specific values |
| **PHYSICS.md** | Complete physics reference | Understanding system in depth |
| **PHYSICS_CONFIG_GUIDE.md** | Parameter tuning | Optimizing physics behavior |
| **PHYSICS_ENGINE_UPGRADES.md** | 250 Hz, restitution, friction cones | Understanding latest improvements |
| **QUADRUPED_BALANCE_TASK.md** | Quadruped design, kinematics | Quadruped morphology details |
| **JOINT_CONSTRAINTS.md** | Joint types, limits, examples | Realistic joint configuration |
| **AGENT_SYSTEM.md** | Neural networks, control | Agent architecture |
| **ARCHITECTURE.md** | 5-class modular design | System structure |
| **WORLD_MODEL.md** | DreamerV3 world model | Training world models |
| **DRL_PHYSICS_PLUGIN.md** | DRL physics integration | Agent-physics interaction |
| **ISAAC_GYM_COMPARISON.md** | Isaac Gym comparison | Industry standard benchmarking |
| **ISAACLAB_COMPARABILITY.md** | IsaacLab compatibility | Modern framework comparison |
| **ACTUATOR_LAG.md** | Motor response time | Motor dynamics tuning |
| **FRICTION_MODELS.md** | Friction models | Ground interaction physics |
| **ENERGY_TRACKING.md** | Power consumption | Energy efficiency metrics |
| **PERFORMANCE.md** | Optimization & benchmarks | Performance analysis |
| **EVALUATION.md** | Evaluation metrics | Testing procedures |
| **INDEX.md** | This file | Documentation navigation |

## 🎯 Where to Start

**Completely new?** → [QUICKSTART.md](QUICKSTART.md) (5 min)  
**Want to understand physics?** → [PHYSICS.md](PHYSICS.md) (comprehensive, 13 sections)  
**Need to configure something?** → [CONFIGURATION.md](CONFIGURATION.md)  
**Want joint constraints?** → [JOINT_CONSTRAINTS.md](JOINT_CONSTRAINTS.md) (jump to "Examples & Usage")  
**Comparing to industry standard?** → [ISAAC_GYM_COMPARISON.md](ISAAC_GYM_COMPARISON.md)

## 🔗 Related Resources

- **Main README**: [../README.md](../README.md) - Project overview
- **Source Code**: [../source/](../source/) - Core implementation
- **Config**: [../config.py](../config.py) - All parameters
- **Training**: [../train.py](../train.py) - Start here to train the agent

## ⚙️ Technical Notes

### Latest Features (Feb 2026)
- **Agent-Centered World Coordinates**: Agent always at origin (0, 0, 0), world is relative
  - Benefits: Translation invariance, better generalization, simplified learning
  - Implementation: Automatic in PhysicsEngine.apply_motor_torques() and Environment.observe()

### Physics Engine Improvements
- **Contact-Dependent Gravity**: `gravity_factor = 1.0 - num_contacts/4.0`
  - Applies full gravity when falling, zero when fully supported
  - Enables natural stance and balance learning
- **Spring-Damper Contact Model**: Realistic foot-ground interactions
  - Parameters: CONTACT_STIFFNESS=0.5, CONTACT_DAMPING=0.2
  - Prevents foot penetration, dissipates energy
- **Joint Velocity Clamping**: Limits to ±10 rad/s for stability
  - Prevents unrealistic motor speeds
  - Matches real actuator constraints
- **Continuous Contact Detection**: Smooth [0, 1] values instead of binary
  - Allows gradient flow for policy learning
  - Better for smooth gait learning

### Architecture
- **5-Class Modular Design** (Updated Feb 2026):
  1. `Environment` - World state, creatures, goals, multi-environment support
  2. `PhysicsEngine` - Motor torques, gravity integration, contacts, balance rewards
  3. `Renderer` - PyRay 3D visualization
  4. `TrainingEngine` - PPO + DreamerV3 world model
  5. `System` - Lightweight orchestrator
  - Benefits: Modular, testable, easy to extend

### Observation & Action Spaces
- **Observation**: 37D (agent-centered coordinates)
  - 12D: Joint angles, 12D: Joint velocities, 4D: Foot contacts
  - 3D: Body orientation, 3D: COM (zeros), 3D: Goal-relative
- **Action**: 12D motor torques for all joints
  - Range: [-5, 5] N⋅m per motor
  - Applied as: joint_vel += (motor_torque - damping * joint_vel) * dt

## 📞 Support

For issues:
1. Check [QUICKSTART.md](QUICKSTART.md)
2. Review [CONFIGURATION.md](CONFIGURATION.md) 
3. Inspect [training.log](../training.log)
