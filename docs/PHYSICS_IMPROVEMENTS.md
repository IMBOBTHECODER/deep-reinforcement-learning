# Physics Engine Improvements Summary

## Architecture: Realism-First with DRL Plugin

The physics engine is now structured as a **self-contained, realistic simulation** with the DRL agent as an optional plugin on top. This philosophy ensures:

1. **Physical Accuracy**: All interactions derive from real-world physics principles
2. **Modularity**: Physics improvements don't require DRL changes
3. **Transferability**: Behaviors learned in simulation are biomechanically valid
4. **Evolution**: Can improve physics without breaking agent training

```
Physical Simulation (PRIMARY - realism, stability, accuracy)
    ├─ Rigid body dynamics (quaternions, inertia, Euler equations)
    ├─ Joint actuators (motors with response lag)
    ├─ Contact mechanics (spring-damper with friction)
    ├─ Ground interaction (Coulomb + viscous friction)
    └─ Energy tracking (power consumption accounting)
        ↓
Observation Space (what agent perceives)
    ├─ Joint angles/velocities (from physics integration)
    ├─ Foot contacts (from contact detection)
    ├─ Body orientation (from quaternion state)
    └─ Goal relative position (from world state)
        ↓
DRL Agent (PLUGIN - learns from observations)
    ├─ Policy network (outputs motor commands)
    ├─ Value network (estimates state value)
    └─ Learning algorithm (PPO + GAE)
```

---

## Phase 1: Actuator Response Lag (IMPLEMENTED ✅)

### Motivation
Real servo motors cannot change output torque instantaneously. They have response lag (1-50ms typical).

### Implementation
**First-Order Lag Model**:
```
τ_applied(t+dt) = τ_applied(t) + (τ_commanded - τ_applied(t)) * (dt / τ_response)
```

Where:
- `τ_response` = response time (seconds, configurable)
- `dt` = physics timestep (0.01s = 10ms)
- This creates realistic servo behavior: commands ramp up, don't snap instantly

### Configuration
```python
# In config.py:
ACTUATOR_RESPONSE_TIME = 0.01  # 10ms lag (typical for servos)
# Set to 0.0 to disable (direct torque application)
```

### Benefits
- **Prevents bang-bang control**: Agent can't apply extreme torques instantly
- **Smooth motion**: Forces realistic smooth acceleration profiles
- **Transfer potential**: Behaviors learned with lag might transfer to real robots better

### Code Location
- [source/physics.py](source/physics.py#L365-L375) - Actuator lag applied before joint dynamics

---

## Phase 2: Improved Friction Model (IMPLEMENTED ✅)

### Motivation
Simple single-coefficient friction is unrealistic. Real friction has:
- **Static coefficient** (higher, prevents slipping)
- **Kinetic coefficient** (lower, reduces during sliding)
- **Viscous damping** (proportional to velocity)

### Implementation
**Coulomb + Viscous Model**:
```
F_friction = {
    μ_static * N                    (if v_slip < threshold)
    μ_kinetic * N + η * v_slip      (if v_slip >= threshold)
}
```

Where:
- `N` = normal contact force
- `v_slip` = horizontal foot velocity
- `η` = viscous damping coefficient

### Configuration
```python
# In config.py:
FRICTION_MODEL = "coulomb+viscous"
FRICTION_COEFFICIENT_STATIC = 0.9   # Higher (prevents initial slip)
FRICTION_COEFFICIENT_KINETIC = 0.85  # Lower (during sliding)
FRICTION_VISCOUS_DAMPING = 0.05     # Velocity-dependent term
FRICTION_SLIP_VELOCITY_THRESHOLD = 0.01  # Transition point
```

### Benefits
- **Realistic stick-slip behavior**: Feet stick when stationary, slip when moving
- **Better balance**: Static friction helps maintain footing
- **Stable gaits**: Reduces unrealistic sliding during locomotion

### Code Location
- [source/physics.py](source/physics.py#L424-L448) - Friction force computation
- [source/physics.py](source/physics.py#L457-L474) - `_compute_friction_force_coulomb_viscous()` method

---

## Phase 4: Energy Tracking (IMPLEMENTED ✅)

### Motivation
Tracking energy consumption enables:
- Efficiency-based rewards (encourage parsimonious motion
- Energy budgeting (limit total activity per episode)
- Realism validation (compare to biological data)

### Implementation
**Mechanical to Electrical Power Conversion**:
```
P_mechanical = Σ(|τ_i| * |ω_i|)  [sum over all joints]
P_electrical = P_mechanical / efficiency
E_total = ∫ P_electrical dt
```

### Configuration
```python
# In config.py:
TRACK_ENERGY_CONSUMPTION = True   # Enable tracking
MOTOR_EFFICIENCY = 0.80            # Typical servo efficiency (60-90%)
```

### Benefits
- **Energy-aware rewards**: Can penalize inefficient behaviors
- **Realistic constraints**: Limited power budget enforces efficient gaits
- **Diagnostics**: Track per-episode energy consumption for analysis

### Code Location
- [source/physics.py](source/physics.py#L398-L415) - Energy tracking in joint updates
- [source/physics.py](source/physics.py#L490-L500) - Energy penalty in reward computation

---

## Summary of Implementation Details

| Feature | Status | Config | Impact |
|---------|--------|--------|--------|
| Quaternion orientation | ✅ Existing | N/A | No gimbal lock |
| Rigid body inertia | ✅ Existing | `BODY_DIMENSIONS` | Realistic mass distribution |
| Spring-damper contacts | ✅ Existing | `CONTACT_STIFFNESS`, `CONTACT_DAMPING` | Foot-ground momentum transfer |
| **Actuator response lag** | ✅ NEW | `ACTUATOR_RESPONSE_TIME` | Realistic servo dynamics |
| **Coulomb+viscous friction** | ✅ NEW | `FRICTION_*` params | Realistic stick-slip behavior |
| **Energy tracking** | ✅ NEW | `TRACK_ENERGY_CONSUMPTION`, `MOTOR_EFFICIENCY` | Efficiency-aware learning |

---

## Backward Compatibility

All new features are **opt-in via configuration**:

```python
# Disable all improvements (legacy behavior):
ACTUATOR_RESPONSE_TIME = 0.0  # No lag
FRICTION_MODEL = "simple"      # Single coefficient
TRACK_ENERGY_CONSUMPTION = False  # No tracking

# Enable all improvements (realism mode):
ACTUATOR_RESPONSE_TIME = 0.01
FRICTION_MODEL = "coulomb+viscous"
TRACK_ENERGY_CONSUMPTION = True
```

**DRL Agent**: Completely **oblivious** to these physics improvements. The agent receives the same observation space (34D) regardless of which physics features are enabled.

---

## Testing & Validation

### Unit Tests (Physics Correctness)
```python
# Verify actuator response is monotonic
t = 0; τ_applied = 0; τ_cmd = 5.0
for i in range(10):
    τ_applied += (τ_cmd - τ_applied) * (0.01 / 0.01)  # Should approach 5.0
    assert τ_applied <= τ_cmd  # Never overshoot

# Verify friction forces are reasonable
F_friction = 0.9 * 49  # μ * mg ≈ 44 N (reasonable for quadruped)
assert F_friction > 0
```

### Integration Tests (Agent Learning)
- [ ] Agent learns faster with realistic physics (lag, friction) than simplified
- [ ] Emergent gaits show realistic patterns (trotting, bounding)
- [ ] Energy consumption correlates with activity level

### Biomechanical Validation
- [ ] Learned motion patterns match natural quadruped gaits
- [ ] Joint torques remain within biological plausibility bounds
- [ ] Balance recovery uses realistic strategies

---

## Future Phases (Planned, Not Yet Implemented)

### Phase 3: Terrain Variation
```python
# Per-region properties:
terrain_stiffness[x,y] = 500 (concrete) or 100 (grass)
terrain_friction[x,y] = 0.9 (rubber) or 0.4 (ice)
terrain_damping[x,y] = 0.15 (soft) or 0.05 (hard)
```
**Impact**: Environment variation requires adaptive gait control.

### Phase 5: Soft Contact Dynamics
```python
# Model foot deformation:
foot_compression = penetration
contact_area = foot_area * (1 + compression_factor * penetration)
pressure = normal_force / contact_area
```
**Impact**: Realistic pressure distributions affect grip and stability.

### Phase 6: Thermal Dynamics
```python
# Motor heating:
T_motor = baseline + integral(I²*R dt)
max_torque = nominal_torque * (1 - 0.01 * (T - T_baseline))
```
**Impact**: Sustained high-torque actions cause servo heating and power loss.

---

## Philosophy Summary

> **"Physics First, DRL Second"**

The physics engine is the foundation. Everything else is built on top:
- Observations derive from physics state
- Rewards measure physical quantities
- Agent learns from realistic dynamics

This ensures that:
✅ Agent learns transferable skills (potentially applicable to real hardware)
✅ Improvements to physics automatically improve all DRL variants
✅ Modular architecture allows physics and learning to evolve independently
✅ Validation is straightforward (compare to physics textbooks or real-world data)

---

## References

### Physics Equations
- **Rigid body dynamics**: [Goldstein, Classical Mechanics](https://en.wikipedia.org/wiki/Classical_mechanics)
- **Quaternion rotations**: [Wikipedia: Quaternions and spatial rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
- **Contact mechanics**: [Hertz contact theory](https://en.wikipedia.org/wiki/Contact_mechanics)
- **Friction models**: [Coulomb friction](https://en.wikipedia.org/wiki/Friction#modelsCoulomb) + viscous damping

### Configuration Parameters
All physics parameters are documented in [config.py](config.py) with inline comments.

### Implementation
- [source/physics.py](source/physics.py) - Physics engine core
- [source/entity.py](source/entity.py) - Creature state + forward kinematics
- [docs/PHYSICS_ENGINE_DESIGN.md](docs/PHYSICS_ENGINE_DESIGN.md) - Architecture overview

