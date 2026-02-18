# Physics Engine Design - Realism First, DRL Plugin

## Architecture Philosophy

The physics engine is the **primary simulation** for accurate quadruped dynamics. The DRL agent is a **plugin** that learns from this realistic simulation. This design ensures:

1. **Realism First**: Physics engine provides ground truth for biomechanics
2. **Transferable Skills**: Agent learns behaviors that could potentially transfer to real robots
3. **Modular**: DRL can be swapped, improved, or compared without changing physics
4. **Interpretability**: All rewards derive from measurable physical quantities
5. **Optimization Path**: Physics improvements directly benefit all DRL variants

### Design Invariant
```
Physical Reality (rigid bodies, contacts, actuators)
    ↓ 
Observation Space (what agent can sense)
    ↓
DRL Agent (learns policy from observations)
    ↓
Action Space (motor controls)
    ↓
Physics Integration (applies actions to dynamics)
```

**Never let DRL requirements simplify physics — instead, improve DRL to work with realistic physics.**

---

## Current Physics Components

### 1. Rigid Body Dynamics
**Status**: ✅ Mature
- Quaternion-based orientation (gimbal-lock-free)
- Full inertia tensor (anisotropic mass distribution)
- Euler equations with gyroscopic effects
- Semi-implicit Euler integration (stable for stiff systems)

### 2. Joint Actuators
**Status**: ⚠️ Oversimplified
- **Current**: Direct torque application with damping
- **In Realism Hierarchy**:
  1. Motor response time (1-2ms lag)
  2. Torque saturation (max torque per joint)
  3. Velocity-dependent friction
  4. Thermal dynamics (actuator heating)

### 3. Contact Model
**Status**: ✅ Basic but working
- Spring-damper (Hooke contact model)
- Unilateral constraint (no adhesion)
- Per-foot contact detection

**Can Improve**:
- Softer contact dynamics (foot deformation)
- Orientation-dependent contact normals
- Separate normal/tangential friction

### 4. Ground Interaction
**Status**: ⚠️ Unrealistic
- Uniform flat ground
- Single friction coefficient
- No terrain deformation
- No rolling resistance

**Can Improve**:
- Friction model: Coulomb + viscous
- Slip behavior: stick-slip transitions
- Spring compliance of ground
- Terrain variation

### 5. Energy/Efficiency
**Status**: ❌ Not modeled
- No actuator power consumption
- No mechanical efficiency losses
- No passive dissipation (air resistance, bearing friction)

**Should Add**:
- Electrical power: P = τ × ω (mechanical power)
- Conversion efficiency: 60-85% for typical actuators
- Resting metabolic cost

---

## Realism Improvement Roadmap

### Phase 1: Actuator Dynamics (Critical)
**Objective**: Make action response more realistic

```
DRL Output (target torque)
    ↓
Actuator Control Loop [0.01-0.05s lag]
    ↓
Motor Response (ramp up)
    ↓
Joint Torque Applied
```

**Implementation**:
```python
# First-order actuator response: τ_actual = τ_actual + (τ_commanded - τ_actual) * (dt / τ_response_time)
self.joint_actuator_response_time = 0.01  # seconds (10ms)
```

**Why Important**:
- Real servos cannot change torque instantly
- Creates realistic "lag" that agents must learn to compensate
- Prevents bang-bang control solutions

### Phase 2: Friction Model (Important)
**Objective**: More accurate foot-ground interaction

```
Friction Force = f(v, N, μ_s, μ_k)
  where:
    v = slip velocity
    N = normal force
    μ_s = static friction (high, prevents slip)
    μ_k = kinetic friction (lower, occurs during slip)
```

**Current**: Simple Coulomb
**Proposed**: Coulomb + viscous damping

```python
# Coulomb friction: F_friction = μ * N
# Viscous damping: F_damping = η * v
F_friction = μ_static * N  # stick if |v| < threshold
F_friction = μ_kinetic * N + η * v  # slip with viscous damping
```

### Phase 3: Terrain Variation (Nice to Have)
**Objective**: Terrain properties affect balance

```python
# Per-region friction coefficients
# Per-region stiffness/compliance
# Per-region surface normal variation
```

### Phase 4: Energy Accounting (Strategic)
**Objective**: Track efficiency, enable efficiency-based rewards

```python
electrical_power = motor_current * motor_voltage
mechanical_power = torque * angular_velocity  
efficiency = mechanical_power / electrical_power
total_energy = integral(electrical_power, dt)
```

---

## Physics Equations Reference

### Rigid Body Dynamics
**Position Integration**:
$$\mathbf{v}_{t+dt} = \mathbf{v}_t + \mathbf{a}_t \cdot dt$$
$$\mathbf{r}_{t+dt} = \mathbf{r}_t + \mathbf{v}_{t+dt} \cdot dt  \quad \text{(semi-implicit Euler)}$$

**Acceleration** (Newton's 2nd Law):
$$\mathbf{a} = \frac{\mathbf{F}}{m}$$

**Angular Motion** (Euler Equations):
$$\mathbf{I} \cdot \boldsymbol{\alpha} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I} \cdot \boldsymbol{\omega})$$

where $\boldsymbol{\omega} \times (\mathbf{I} \cdot \boldsymbol{\omega})$ is the gyroscopic torque.

**Quaternion Integration**:
$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes [0, \boldsymbol{\omega}]$$
$$\mathbf{q}_{t+dt} = \mathbf{q}_t + \dot{\mathbf{q}}_t \cdot dt  \quad \text{(normalized)}$$

### Contact Model
**Spring-Damper (Unilateral)**:
$$F_{contact} = \max(0, k \cdot d + c \cdot v)$$

where:
- $d$ = penetration depth
- $v$ = normal velocity (into ground)
- $k$ = contact stiffness
- $c$ = contact damping

### Friction Model
**Coulomb Friction** (static/kinetic):
$$F_{friction} = \mu \cdot N$$

**Viscous Damping**:
$$F_{damping} = \eta \cdot v_{slip}$$

**Combined**:
$$F_{frictional} = \min(\mu \cdot N, \sqrt{\tau^2}) + \eta \cdot v_{slip}$$

### Joint Actuator Dynamics
**Stiff Damping Model**:
$$\tau_{applied} = \tau_{commanded} - d \cdot \omega$$

**First-Order Response** (more realistic):
$$\tau_{applied,t+dt} = \tau_{applied,t} + \frac{(\tau_{commanded} - \tau_{applied,t})}{\tau_{response}} \cdot dt$$

---

## DRL Integration Points

### Observation Space (what DRL sees)
These come **directly from physics**:
```python
observation = [
    joint_angles[12],        # From joint position integration
    joint_velocities[12],    # From joint acceleration integration
    foot_contacts[4],        # From contact detection
    body_orientation[3],     # From quaternion.to_euler()
    body_angular_velocity[3],  # From physics.body.angular_vel
    goal_relative[3],        # From reward computation
]
```

### Action Space (what DRL outputs)
Applied **through physics** (not bypassing it):
```python
motor_torques = policy_network(observation)  # DRL output
motor_torques = clamp(motor_torques, -5, 5)  # Saturation
motor_torques = apply_actuator_response(motor_torques)  # Lag
physics_engine.apply_motor_torques(creature, motor_torques)
```

### Reward Signals (derived from physics)
All rewards computed from **observed physical state**:
```python
balance_reward = measure_stability(body_orientation)
contact_reward = measure_foot_contacts(foot_contact_sensors)
progress_reward = measure_goal_distance(com_pos, goal)
efficiency_penalty = measure_energy_consumption(motor_torques, joint_vels)
```

---

## Validation & Testing

### Unit Tests (Physics Correctness)
- [ ] Quaternion normalization preserves unit length
- [ ] Inertia tensor computation matches rigid body theory
- [ ] Contact force unilaterality (no adhesion)
- [ ] Energy conservation (no spurious gains/losses)

### Integration Tests (Agent Learning)
- [ ] Agent learns faster on realistic physics than simplified
- [ ] Contact patterns match quadruped gait analysis
- [ ] Energy consumption correlates with motor commands

### Benchmark Tests (Real-World Transfer)
- [ ] Gaits learned in simulation look biomechanically plausible
- [ ] Balance recovery strategies match real quadrupeds
- [ ] Efficiency metrics align with biology

---

## Summary: Physics First, DRL Second

| Decision | Why |
|----------|-----|
| **Quaternion orientation** | Prevents gimbal lock; enables smooth 3D rotation |
| **Full inertia tensor** | Models realistic mass distribution |
| **Spring-damper contacts** | Accurate foot-ground momentum transfer |
| **Joint damping** | Realistic actuator friction and response |
| **Direct torque application** | Simplest valid model (can add response lag) |
| **DRL as plugin** | Can swap/upgrade without rebuilding physics |

The physics engine is the **foundation**. Everything else (observations, rewards, agent architecture, training algorithms) is built **on top** of it. This ensures the agent learns from reality, not from simplified models.

