# Physics Configuration Guide

## Quick Start: Presets

Choose a physics configuration based on your goal:

### Preset 1: "Speed Focus" (Baseline, Simplest)
For rapid training and debugging:
```python
ACTUATOR_RESPONSE_TIME = 0.0           # No lag, instant response
FRICTION_MODEL = "simple"              # Single friction coefficient
TRACK_ENERGY_CONSUMPTION = False       # Minimal overhead
DT = 0.01                              # Standard 100 Hz
```
**When to use**: Testing DRL algorithm changes quickly
**Training speed**: Fastest
**Realism**: Lowest

---

### Preset 2: "Balanced" (Recommended Default)
Medium realism with reasonable training speed:
```python
ACTUATOR_RESPONSE_TIME = 0.01          # 10ms lag (typical servo)
FRICTION_MODEL = "coulomb+viscous"     # Realistic stick-slip
TRACK_ENERGY_CONSUMPTION = True        # Monitor efficiency
DT = 0.01                              # 100 Hz physics
```
**When to use**: Most training runs, research studies
**Training speed**: Normal
**Realism**: High

---

### Preset 3: "Realism Focus" (Simulation Research)
Maximum physical accuracy:
```python
ACTUATOR_RESPONSE_TIME = 0.02          # 20ms lag (slower servo)
FRICTION_MODEL = "coulomb+viscous"     # Full friction model
TRACK_ENERGY_CONSUMPTION = True        # Full energy tracking
FRICTION_COEFFICIENT_STATIC = 0.95     # High grip
FRICTION_COEFFICIENT_KINETIC = 0.80    # Realistic slip
DT = 0.005                             # 200 Hz (finer granularity)
```
**When to use**: Validation, real-world comparison, hardware transfer
**Training speed**: Slower
**Realism**: Maximum

---

## Parameter Tuning Guide

### Actuator Response Time
Controls how quickly motors respond to commands.

```python
ACTUATOR_RESPONSE_TIME = 0.01  # seconds
```

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Instant response (no lag) | Baseline, reference |
| 0.005 | Very fast (5ms) | Aggressive control |
| 0.01 | Standard (10ms) | Typical servo motors |
| 0.02 | Slow (20ms) | Older/heavier servos |
| 0.05 | Very slow (50ms) | Heavy duty motors |

**How it affects agents**:
- **Low lag** (0.0-0.005): Agent can make sharp motion changes, jerky gaits possible
- **Medium lag** (0.01): Agent learns smooth, realistic motions
- **High lag** (0.05+): Agent must plan movements in advance, smooth gaits required

**Tuning tip**: Increase if agent learns unrealistic bang-bang control

---

### Friction Model
Determines how feet interact with ground.

```python
FRICTION_MODEL = "coulomb+viscous"  # Options: "simple", "coulomb", "coulomb+viscous"
```

#### Simple Friction (Legacy)
```python
FRICTION_MODEL = "simple"
GROUND_FRICTION_COEFFICIENT = 0.9  # Single value
```
**Formula**: F = μ × N
**Pros**: Fast, simple, well-understood
**Cons**: Unrealistic static/kinetic difference, no velocity dependence

#### Coulomb Friction
```python
FRICTION_MODEL = "coulomb"
FRICTION_COEFFICIENT_KINETIC = 0.85
```
**Formula**: F = μ_k × N
**Pros**: Kinetic friction (realistic for sliding)
**Cons**: No static friction, no viscous damping

#### Coulomb + Viscous (Recommended)
```python
FRICTION_MODEL = "coulomb+viscous"
FRICTION_COEFFICIENT_STATIC = 0.9
FRICTION_COEFFICIENT_KINETIC = 0.85
FRICTION_VISCOUS_DAMPING = 0.05
FRICTION_SLIP_VELOCITY_THRESHOLD = 0.01
```
**Formula**: 
- If v_slip < threshold: F = μ_s × N
- If v_slip ≥ threshold: F = μ_k × N + η × v_slip

**Pros**: Realistic stick-slip behavior, smooth transitions
**Cons**: More compute (negligible), four parameters to tune

### Friction Parameter Details

#### Static vs Kinetic Coefficient
```python
FRICTION_COEFFICIENT_STATIC = 0.9   # Higher: harder to start slipping
FRICTION_COEFFICIENT_KINETIC = 0.85  # Lower: easier to continue sliding
```

**Real-world values**:
- Rubber on concrete: μ_s ≈ 0.9, μ_k ≈ 0.7
- Animal pads on ground: μ_s ≈ 0.8-0.95, μ_k ≈ 0.65-0.85
- Bio-inspired pads: μ_s ≈ 1.0, μ_k ≈ 0.8

**Effect on agent**:
- High static (0.95): Feet stick well, harder to start movement
- Low kinetic (0.70): Feet slide easily once moving, less grip

#### Viscous Damping
```python
FRICTION_VISCOUS_DAMPING = 0.05  # N·s/m
```
**Formula**: Adds velocity-proportional resistance

**Effect**:
- Low (0.0): No velocity damping, unrealistic sliding
- Medium (0.05): Realistic damping, smooth deceleration
- High (0.2): Heavy damping, sluggish motion

#### Slip Threshold
```python
FRICTION_SLIP_VELOCITY_THRESHOLD = 0.01  # m/s
```
**Transition point** between static and kinetic friction.

**Effect**:
- Too low (0.001): Immediate kinetic friction, slippery
- Appropriate (0.01): Realistic transition
- Too high (0.1): Sticky grip, unrealistic

---

### Energy Tracking
Monitor and penalize inefficiency.

```python
TRACK_ENERGY_CONSUMPTION = True
MOTOR_EFFICIENCY = 0.80
ENERGY_PENALTY = 0.01  # In reward function
```

#### Motor Efficiency
```python
MOTOR_EFFICIENCY = 0.80  # 80% mechanical efficiency
```

**Real-world typical values**:
- Servo motors: 60-90%
- DC motors: 75-90%
- Brushless motors: 85-95%

**How it affects learning**:
- High efficiency (0.95): Electric power ≈ mechanical power, wasteful actions cost less
- Medium efficiency (0.80): Realistic servo behavior
- Low efficiency (0.60): Heavy cost to high-power actions

#### Tracked Energy Reporting
When enabled, each simulation step reports:
```python
stability_metrics['energy_consumed']  # Joules per step
```

**Usage**:
```python
total_energy = sum(energy_per_step)
print(f"Episode energy: {total_energy:.1f} J")

energy_per_meter = total_energy / distance_traveled
print(f"Efficiency: {energy_per_meter:.2f} J/m")
```

---

## Contact Physics Parameters

### Contact Stiffness
```python
CONTACT_STIFFNESS = 500.0  # N/m per foot
```

Higher = stiffer ground (concrete-like)
Lower = softer ground (grass-like)

| Value | Surface | Effect on agent |
|-------|---------|-----------------|
| 100 | Soft sand | Sinking, less rebound |
| 300 | Grass | Natural compliance |
| 500 | Concrete | Firm, reactive |
| 1000 | Steel | Very hard, bouncy |

### Contact Damping
```python
CONTACT_DAMPING = 0.15
```

Controls impact absorption.

| Value | Type | Effect |
|-------|------|--------|
| 0.0 | Elastic | Bouncing, unstable |
| 0.1 | Normal | Moderate damping |
| 0.3 | Heavy | Very stable, dull |

---

## Diagnostics & Debugging

### Check Physics Settings
```python
print(f"Actuator lag: {Config.ACTUATOR_RESPONSE_TIME * 1000:.1f} ms")
print(f"Friction model: {Config.FRICTION_MODEL}")
print(f"Energy tracking: {Config.TRACK_ENERGY_CONSUMPTION}")
print(f"Contact stiffness: {Config.CONTACT_STIFFNESS} N/m")
```

### Monitor During Training
```python
# Enable in training loop
if step % 100 == 0:
    print(f"Avg energy/step: {avg_energy:.3f} J")
    print(f"Foot contact count: {sum(num_contacts)}")
    print(f"Action clamp fraction: {clamp_fraction:.2%}")
    print(f"Balance: pitch={pitch:.3f}, roll={roll:.3f}")
```

### Common Issues & Solutions

**Issue**: Agent learns jerky, unrealistic motion
```python
# Solution: Increase actuator lag
ACTUATOR_RESPONSE_TIME = 0.01  # Was 0.0
```

**Issue**: Agent feet slip constantly
```python
# Solution: Increase static friction
FRICTION_COEFFICIENT_STATIC = 0.95  # Was 0.9
```

**Issue**: Agent moves but seems inefficient
```python
# Solution: Enable energy tracking
TRACK_ENERGY_CONSUMPTION = True
# And add energy penalty in reward
```

**Issue**: Motion is unrealistically smooth
```python
# Solution: Enable friction viscous damping
FRICTION_VISCOUS_DAMPING = 0.05
```

**Issue**: Physics simulation is too slow
```python
# Solution: Reduce fidelity
TRACK_ENERGY_CONSUMPTION = False  # Disable heavy tracking
DT = 0.02  # Slower simulation (50Hz instead of 100Hz)
```

---

## Advanced: Custom Physics Scenarios

### Scenario 1: Icy Surface
```python
FRICTION_COEFFICIENT_STATIC = 0.3   # Low grip
FRICTION_COEFFICIENT_KINETIC = 0.2  # Super slippery
FRICTION_VISCOUS_DAMPING = 0.01     # Minimal damping
```
**Result**: Agent must learn careful, constrained movements

### Scenario 2: Muddy Terrain
```python
CONTACT_STIFFNESS = 150.0   # Soft ground
CONTACT_DAMPING = 0.3       # Heavy damping
FRICTION_MODEL = "coulomb"  # High drag
FRICTION_COEFFICIENT_KINETIC = 1.0  # Very high friction
```
**Result**: Agent learns to lift feet high, energetic gait

### Scenario 3: Precision Control (Robot Transfer)
```python
ACTUATOR_RESPONSE_TIME = 0.02  # Real servo lag
FRICTION_MODEL = "coulomb+viscous"
TRACK_ENERGY_CONSUMPTION = True
MOTOR_EFFICIENCY = 0.75  # Real servo efficiency
CONTACT_STIFFNESS = 800.0  # Firm ground
```
**Result**: Behaviors likely to transfer to real hardware

### Scenario 4: Efficiency Focus
```python
TRACK_ENERGY_CONSUMPTION = True
MOTOR_EFFICIENCY = 0.70  # Penalize inefficiency
# In reward: add high weight to energy penalty
ENERGY_PENALTY = 0.05  # Was 0.01
```
**Result**: Agent optimizes for low-power operation

---

## Summary: Configuration Checklist

Before training, answer these:

- [ ] What's my primary goal? (speed vs realism)
- [ ] Do I need realistic servos? (set ACTUATOR_RESPONSE_TIME)
- [ ] Do I care about foot grip? (set FRICTION_MODEL)
- [ ] Should I track efficiency? (set TRACK_ENERGY_CONSUMPTION)
- [ ] What terrain type? (adjust contact stiffness/damping)
- [ ] Do I plan real-world deployment? (use Preset 3)

**Recommendation**: Start with Preset 2 (Balanced), adjust based on results.

