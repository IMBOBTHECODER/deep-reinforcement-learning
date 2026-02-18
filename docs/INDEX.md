# Quadruped Balance System - Documentation Index

Welcome! This folder contains comprehensive documentation for the agent-centered quadruped balance task with advanced physics simulation.

## 📚 Documentation Structure

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in 5 minutes
- **[CONFIGURATION.md](CONFIGURATION.md)** - All configuration parameters explained

### Latest Updates (February 2026)
- **[PHYSICS_ENGINE_UPGRADES.md](PHYSICS_ENGINE_UPGRADES.md)** - ⭐ NEW: 250 Hz, restitution, friction cones, vectorization
- **[ISAAC_GYM_COMPARISON.md](ISAAC_GYM_COMPARISON.md)** - How we compare to NVIDIA Isaac Gym standard
- **[QUADRUPED_BALANCE_TASK.md](QUADRUPED_BALANCE_TASK.md)** - Quadruped architecture, 12 DOF forward kinematics, balance-focused rewards
- **[PHYSICS_AND_WORLD.md](PHYSICS_AND_WORLD.md)** - Improved physics (gravity integration, spring-damper contacts, velocity clamping, agent-centered world)

### Core Systems & Agent Design
- **[AGENT_SYSTEM.md](AGENT_SYSTEM.md)** - Agent neural networks, policy/value heads, motor control
- **[WORLD_SYSTEM.md](WORLD_SYSTEM.md)** - Physics engine, reward function, environment mechanics

### Physics & Dynamics Runtime
- **[PHYSICS_QUICK_START.md](PHYSICS_QUICK_START.md)** - **START HERE** - 3 presets, copy-paste configs
- **[PHYSICS_FUNDAMENTALS.md](PHYSICS_FUNDAMENTALS.md)** - Core concepts, architecture, observation/action spaces
- **[ACTUATOR_LAG.md](ACTUATOR_LAG.md)** - Motor response time tuning (Phase 1)
- **[FRICTION_MODELS.md](FRICTION_MODELS.md)** - Stick-slip friction comparison (Phase 2)  
- **[ENERGY_TRACKING.md](ENERGY_TRACKING.md)** - Power consumption monitoring (Phase 4)
- **[PHYSICS_DELIVERY_SUMMARY.md](PHYSICS_DELIVERY_SUMMARY.md)** - What's included, learning paths, FAQ

### Physics Theory & Design
- **[PHYSICS_ENGINE_DESIGN.md](PHYSICS_ENGINE_DESIGN.md)** - Design philosophy, roadmap (Phases 1-6)
- **[DRL_PHYSICS_PLUGIN.md](DRL_PHYSICS_PLUGIN.md)** - How DRL integrates with physics engine

### Architecture & Deep Dives
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, 5-class modular architecture, data flow
- **[WORLD_MODEL.md](WORLD_MODEL.md)** - DreamerV3 world model training
- **[PERFORMANCE.md](PERFORMANCE.md)** - Optimization, kernel fusion, profiling

## 🎯 Quick Links by Use Case

### "I want to start training NOW"
→ [PHYSICS_QUICK_START.md](PHYSICS_QUICK_START.md) (2 min, copy preset, done!)

### "I want to understand the physics system"
→ [PHYSICS_FUNDAMENTALS.md](PHYSICS_FUNDAMENTALS.md) (overview) → [PHYSICS_ENGINE_DESIGN.md](PHYSICS_ENGINE_DESIGN.md) (architecture)

### "I want to tune for a specific use case"
→ [PHYSICS_QUICK_START.md](PHYSICS_QUICK_START.md) (presets) → [ACTUATOR_LAG.md](ACTUATOR_LAG.md) / [FRICTION_MODELS.md](FRICTION_MODELS.md) / [ENERGY_TRACKING.md](ENERGY_TRACKING.md)

### "What exactly was delivered?"
→ [PHYSICS_DELIVERY_SUMMARY.md](PHYSICS_DELIVERY_SUMMARY.md) (complete inventory)

## 📋 File Map

| Document | Topics |
|----------|--------|
| QUICKSTART.md | Setup, run, first training results |
| CONFIGURATION.md | All parameters, physics tuning, reward weights |
| **QUADRUPED_BALANCE_TASK.md** | **Quadruped design, 4 legs × 3 joints, forward kinematics, balance task** |
| **PHYSICS_AND_WORLD.md** | **Gravity integration, spring-damper contacts, joint velocity limits, agent-centered world** |
| AGENT_SYSTEM.md | Neural networks, motor control, policy/value heads |
| WORLD_SYSTEM.md | Physics engine, reward computation, mechanics |
| ARCHITECTURE.md | 5-class modular design (Environment, PhysicsEngine, Renderer, TrainingEngine, System) |
| WORLD_MODEL.md | DreamerV3 world model, dynamics learning |
| PERFORMANCE.md | Optimization, JIT kernel fusion, profiling |

**Total: ~4000 lines of comprehensive documentation**

## 🔗 Related Resources

- **Main README**: [../README.md](../README.md) - Project overview
- **Source Code**: [../source/](../source/) - Implementation
- **Config**: [../config.py](../config.py) - Configuration file

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
