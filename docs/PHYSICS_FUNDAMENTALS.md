# Physics Fundamentals - Quickscan Reference

**Need a quick overview of how the physics engine works?** This page has the essentials.

---

## System Architecture

```
┌──────────────────────────────────────┐
│  DRL Agent (12D motor commands)      │
└───────────────────┬──────────────────┘
                    │ motor_torques[12]
                    ↓
┌──────────────────────────────────────┐
│  Physics Engine (Primary Simulation) │
│  ├─ Actuator Dynamics (optional lag) │
│  ├─ Joint Kinematics                 │
│  ├─ Rigid Body Dynamics              │
│  ├─ Contact Detection                │
│  ├─ Friction Forces                  │
│  └─ Energy Tracking (optional)       │
└───────────────────┬──────────────────┘
                    │ observations[34], reward, done
                    ↓
```

**Key Principle**: Physics is PRIMARY (realistic). DRL is a PLUGIN that learns from it.

---

## Core Physics Components

### 1. Rigid Body Dynamics
**What**: The quadruped's body as a rigid object in 3D space

```
Position:     x, y, z (where is the body?)
Velocity:     vx, vy, vz (how fast is it moving?)
Orientation:  quaternion (which way is it facing?)
Angular Vel:  wx, wy, wz (spinning how fast?)
```

**Equations**:
- Position: `x += v * dt`
- Velocity: `v += a * dt` where `a = F/m`
- Rotation: `q += 0.5 * q * ω * dt` (quaternion integration)

✅ **Status**: Excellent, Euler equations with gyroscopic effects

---

### 2. Joint Actuators (12 joints × 4 legs)
**What**: Motors that apply torques to move joints

```
Command:  agent outputs (e.g., 2.5 N⋅m)
          ↓ [optional actuator lag]
Applied:  motor actually applies (e.g., ramps to 2.5)
          ↓ [joint equation]
Result:   θ (angle), ω (velocity)
```

**Equations**:
- With lag: `τ_applied = τ_applied + (τ_command - τ_applied) * (dt/τ_response)`
- Without lag: `τ_applied = τ_command` (instant)
- Joint dynamics: `ω += (τ - damping*ω) * dt`

⚙️ **Feature**: Phase 1 - Optional actuator lag (0-50ms)

---

### 3. Contact & Friction
**What**: How feet interact with ground

```
Ground
═════════════════════════════════════
     ↑ Contact!
    Foot

Three friction components:
1. Spring force (ground pushes back)
2. Damping (absorbs impact)
3. Friction (opposes sliding)
```

**Equations**:
- Contact: `F_normal = k*penetration - c*v_z`
- Friction (simple): `F_friction = μ * F_normal`
- Friction (realistic): `F_friction = μ * F_normal + η * v_slip`

📈 **Feature**: Phase 2 - Realistic Coulomb + viscous friction

---

### 4. Energy Tracking
**What**: Monitor power consumption

```
Mechanical Power:  P = Σ|torque × velocity|
Electrical Power:  P_elec = P_mech / efficiency
Energy:            E = ∫ P_elec dt
```

⚡ **Feature**: Phase 4 - Optional energy accounting

---

## Configuration Levels

### Minimal (Baseline)
```python
ACTUATOR_RESPONSE_TIME = 0.0
FRICTION_MODEL = "simple"
TRACK_ENERGY_CONSUMPTION = False
```
→ Fastest training, least realistic

### Realistic (Recommended)
```python
ACTUATOR_RESPONSE_TIME = 0.01
FRICTION_MODEL = "coulomb+viscous"
TRACK_ENERGY_CONSUMPTION = True
```
→ Balanced realism & speed

### Maximum
```python
ACTUATOR_RESPONSE_TIME = 0.02
FRICTION_MODEL = "coulomb+viscous"
FRICTION_VISCOUS_DAMPING = 0.08
TRACK_ENERGY_CONSUMPTION = True
```
→ Highest realism (slowest training)

---

## Observation Space (What Agent Sees)

```python
observation = [
    # Joint state (12 joints × 4 legs)
    angle[0:12],        # Joint angles: [-π, π]
    velocity[0:12],     # Joint velocities: [-10, 10] rad/s
    
    # Foot contact sensors
    contact[0:4],       # Foot in contact? [0.0 or 1.0]
    
    # Body state
    pitch, yaw, roll,   # Body orientation
    
    # Goal (relative to body)
    goal_x, goal_y, goal_z  # Target position
]
# Total: 34D
```

All from physics state directly (no synthetic features).

---

## Action Space (What Agent Controls)

```python
motor_torques = [
    # Each joint gets a torque command
    τ[0:12]  # One for each joint
]
# Range: [-5, 5] N⋅m (clamped)
# Real-time control: sent to physics each 10ms
```

Applied through physics (not bypassing it).

---

## Reward Decomposition

```python
total_reward = (
    # Primary: Stay upright
    balance_reward
    +  contact_reward
    -  tilt_penalty
    
    # Secondary: Reach goal
    +  goal_reward (modulated by balance)
    
    # Tertiary: Be efficient
    -  energy_penalty
)
```

Each term measures observable physical quantities.

---

## Performance Considerations

### Optimization Status ✅

| Component | CPU Cost | GPU? | Disabled By Default? |
|-----------|----------|------|-------------|
| Joint dynamics | Fast | Yes | No (always) |
| Friction | Fast | No | No (always) |
| Actuator lag | Negligible | No | Yes ✅ |
| Energy tracking | Negligible | No | Yes ✅ |
| Contact detection | Fast | No | No (always) |

**Key**: Expensive features (lag, energy) are disabled by default.

### Scaling
- **Single creature**: ~0.5ms per physics step (CPU)
- **8 parallel creatures**: ~2-3ms per step (CPU)
- **GPU acceleration**: 30-50% faster on NVIDIA RTX

---

## Quick Diagnostics

### Check Physics is Working
```python
# After 1 step
print(f"Body position: {creature.pos}")  # Should change
print(f"Joint angles: {creature.joint_angles}")  # Should change
print(f"Foot contacts: {creature.foot_contact}")  # 0-4 contacts

# If all zero or unchanged → problem!
```

### Monitor Agent Interaction
```python
print(f"Motor commands: {motor_torques}")
print(f"Body orientation: {creature.orientation}")
print(f"Energy: {creature._total_energy_consumed:.1f} J")
```

---

## Next: Choose Your Path

### 👤 "I want to start training ASAP"
→ [PHYSICS_QUICK_START.md](PHYSICS_QUICK_START.md) (pick Preset 2, done!)

### 📖 "I want to understand each feature"
→ [ACTUATOR_LAG.md](ACTUATOR_LAG.md) → [FRICTION_MODELS.md](FRICTION_MODELS.md) → [ENERGY_TRACKING.md](ENERGY_TRACKING.md)

### 🏗️ "I want to understand the architecture"
→ [PHYSICS_ENGINE_DESIGN.md](PHYSICS_ENGINE_DESIGN.md) → [DRL_PHYSICS_PLUGIN.md](DRL_PHYSICS_PLUGIN.md)

### 📊 "I want to tune for a specific scenario"
→ Read feature doc above, copy scenario example

---

## Summary

**Physics Engine Provides**:
- ✅ Realistic 3D rigid body dynamics
- ✅ Joint kinematics (forward & inverse)
- ✅ Contact detection and response
- ✅ 3 friction models (pick one)
- ✅ Optional motor lag
- ✅ Optional energy tracking
- ✅ All configuration-driven

**Agent Gets**:
- ✅ 34D observation (all from physics)
- ✅ Reward signal (physical quantities)
- ✅ Realistic environment (no shortcuts)

**Philosophy**: 
> Physics is the primary simulation. The agent learns from reality, not a simplified model.

