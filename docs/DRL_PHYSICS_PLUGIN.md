# DRL as Plugin Architecture

## Separation of Concerns

The quadruped system has a clean **two-layer architecture**:

```
┌─────────────────────────────────────────┐
│  DEEP REINFORCEMENT LEARNING (Plugin)   │
│  ├─ Policy Network (12D output)         │
│  ├─ Value Network (1D output)           │
│  ├─ PPO Training Loop                   │
│  └─ Gradient Computation                │
└────────────────┬────────────────────────┘
                 │ motor_torques[12]
                 ↓
┌─────────────────────────────────────────┐
│  PHYSICS ENGINE (Primary Foundation)    │
│  ├─ Rigid Body Dynamics                 │
│  ├─ Joint Actuators                     │
│  ├─ Contact Detection & Forces          │
│  ├─ Friction (Static/Kinetic/Viscous)   │
│  ├─ Energy Tracking                     │
│  └─ Integration to Next State            │
└────────────────┬────────────────────────┘
                 │ observations[34]
                 ↓
```

## DRL ↔ Physics Interface

### DRL Outputs (What the agent controls)
```python
# From policy_network.forward(observation)
motor_torques: torch.Tensor  # Shape: (batch, 12)
# Range: [-5, 5] N⋅m per joint (clamped)
# Sent to physics engine
```

**These torques are applied to the physics engine, NOT directly to joint angles.**

### Physics Computes
1. **Actuator response lag** (if enabled)
2. **Joint dynamics** (τ → ω, ω → θ via integration)
3. **Forward kinematics** (θ → foot positions)
4. **Contact detection** (foot_z vs ground_level)
5. **Friction forces** (based on foot slip velocity)
6. **Rigid body forces** (gravity + contact reactio)
7. **Integration** (position, velocity, orientation update)

### DRL Inputs (What the agent observes)
```python
observation: torch.Tensor  # Shape: (batch, 34)
# Breakdown:
#   [0:12]   joint_angles          (from physics integration)
#   [12:24]  joint_velocities      (from physics integration)
#   [24:28]  foot_contacts         (from contact detection)
#   [28:31]  body_orientation      (pitch, yaw, roll)
#   [31:34]  goal_relative_pos     (x, z, -y in agent frame)
```

**These observations come DIRECTLY from the physics state, not synthesized.**

---

## Example Flow: Single Simulation Step

```python
# DRL Forward Pass
observation = torch.randn(batch_size, 34)  # From physics
with torch.no_grad():
    policy_input = encoder(observation)     # 34 → 256 → 128
    features = feature_projection(policy_input)  # 128 → 256
    h_t, c_t = lstm(features, prev_state)   # LSTM update
    mu = policy_mu(h_t)                     # → 12D means
    sigma = exp(log_std_param)
    
motor_torques = mu + sigma * randn(batch_size, 12)  # Reparameterization
motor_torques = clamp(motor_torques, -5, 5)  # Saturation

# ↓↓↓ PASS TO PHYSICS ↓↓↓

# Physics Simulation
creature.apply_motor_torques(motor_torques)

# Step 1: Actuator response lag (Phase 1)
if ACTUATOR_RESPONSE_TIME > 0:
    creature._actuator_state += (motor_torques - creature._actuator_state) * (dt / response_time)
    motor_torques_applied = creature._actuator_state

# Step 2: Joint dynamics (core physics)
joint_accelerations = (motor_torques - DAMPING * joint_velocities) / inertia
joint_velocities += joint_accelerations * dt
joint_angles += joint_velocities * dt

# Step 3: Forward kinematics
foot_positions = forward_kinematics(joint_angles)

# Step 4: Contact detection
for foot_idx in range(4):
    if foot_positions[foot_idx].z <= ground_level:
        foot_contacts[foot_idx] = 1.0
        
        # Spring-damper contact
        normal_force = k_contact * penetration - c_contact * v_z
        
        # Phase 2: Friction (Coulomb + viscous)
        friction = compute_friction(normal_force, slip_velocity)
        
        # Apply forces to rigid body
        contact_force += [friction_x, friction_y, normal_force_z]

# Step 5: Rigid body integration
com_accel = contact_force / mass + gravity
com_velocity += com_accel * dt
com_position += com_velocity * dt

# Quaternion integration (orientation)
q_dot = 0.5 * q * omega_quaternion
q = q + q_dot * dt
q.normalize()

# Step 6: Phase 4 - Energy tracking (optional)
if TRACK_ENERGY:
    mechanical_power = sum(|tau_i * omega_i|)
    electrical_power = mechanical_power / efficiency
    total_energy_consumed += electrical_power * dt

# Step 7: Reward computation (uses physical state)
balance_reward = exp(-0.5 * (pitch² + roll²) / 0.1²)
contact_reward = num_contacts * 0.1
energy_penalty = |motor_torques| * 0.01
total_reward = balance_reward + contact_reward - energy_penalty

# ↑↑↑ END PHYSICS ↑↑↑

# Back to DRL: Use observations from physics
new_observation = extract_observation_from_physics(creature)
new_log_prob = compute_log_prob(motor_torques, mu, sigma)
value = value_network(h_t)

# Store in PPO buffer for later training
buffer.store(
    observation=observation,
    action=motor_torques,
    log_prob=new_log_prob,
    reward=total_reward,
    value=value,
    done=episode_finished
)
```

---

## Why This Architecture is Superior

### 1. **Physics Independence**
DRL algorithm changes don't require physics rewrites:
```python
# Can switch to A3C, TRPO, SAC, etc. without touching physics
drl_agent = A3C(policy_net, value_net)  # Physics unchanged
```

### 2. **Physics Improvements Benefit All Agents**
Add new friction model → **all agents learn from it automatically**:
```python
FRICTION_MODEL = "coulomb+viscous"  # Physics change only
# No DRL code modification needed!
```

### 3. **Interpretability**
Every reward term is a measurable physical quantity:
- `balance_reward` = stability (tilt angle)
- `contact_reward` = foot contact count
- `energy_penalty` = torque magnitude

### 4. **Validation**
Compare learned behaviors to:
- Physics textbooks (confirm dynamics correct)
- Real quadrupeds (compare gait patterns)
- Hardware testing (reality check)

### 5. **Modularity**
Can test physics without DRL:
```python
# Pure physics test
creature.apply_motor_torques(torch.zeros(12))  # No action
# Physics runs normally, can measure stability
```

---

## DRL Parameters vs Physics Parameters

### DRL-Controlled (don't affect physics accuracy)
```python
# In config.py
PPO_CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
LR = 3e-4
GAMMA = 0.99
```
These are learning hyperparameters. Changing them doesn't change physics.

### Physics-Controlled (affect simulation accuracy)
```python
# In config.py
ACTUATOR_RESPONSE_TIME = 0.01  # ← Physics realism
FRICTION_MODEL = "coulomb+viscous"  # ← Physics realism
CONTACT_STIFFNESS = 500.0  # ← Physics realism
```
Changing these directly affects what the agent learns from.

---

## How to Modify Physics Without Breaking DRL

### Example 1: Add Ground Elasticity
```python
# physics.py
class PhysicsEngine:
    def _update_joint_dynamics_cpu(self, creature, motor_torques):
        # ... existing code ...
        
        # NEW: Ground elasticity
        if self.track_ground_elasticity:
            restoration_force = -self.ground_elasticity * ground_deformation
            contact_force_total[2] += restoration_force
        
        # ... rest of code unchanged ...
```
**Result**: Agent sees same observations, but physics is more detailed. Learning automatically adapts.

### Example 2: Add Joint Position Limits
```python
# physics.py
creature.joint_angles = torch.clamp(
    creature.joint_angles,
    -math.pi,  # Existing
    math.pi    # Existing
)

# NEW: Hard stop penalties
for i in range(12):
    if creature.joint_angles[i] > MAX_ANGLE[i]:
        # Can add torque penalty or hard contact
        pass
```
**Result**: Agent learns to respect joint limits without explicit code change.

---

## Monitoring DRL ↔ Physics Interaction

### During Training
```python
# Log what agent is actually doing
for step in train_loop:
    motor_commands = agent.forward(obs)
    motor_applied = physics.apply_with_lag(motor_commands)
    
    print(f"Command: {motor_commands[0]:.3f} → Applied: {motor_applied[0]:.3f}")
    print(f"Lag effect: {(1 - motor_applied[0]/motor_commands[0])*100:.1f}%")
```

### Diagnose Issues
```python
# If agent learns jerky motion:
# → Increase ACTUATOR_RESPONSE_TIME (smoother commands)

# If agent slips on ground:
# → Increase FRICTION_COEFFICIENT_STATIC (better grip)

# If agent is inefficient:
# → Enable TRACK_ENERGY_CONSUMPTION (add efficiency reward)
```

---

## Summary

**DRL is a plugin** that learns to control a realistic physics engine:

1. ✅ **Physics first**: Simulation is accurate and self-contained
2. ✅ **DRL oblivious**: Agent doesn't care about physics improvements
3. ✅ **Realism-focused**: Each physics feature has a real-world justification
4. ✅ **Modular evolution**: Physics and learning improve independently
5. ✅ **Transferable**: Learned behaviors are biomechanically plausible

**This architecture enables the system to evolve toward both realism AND performance without artificial constraints.**

