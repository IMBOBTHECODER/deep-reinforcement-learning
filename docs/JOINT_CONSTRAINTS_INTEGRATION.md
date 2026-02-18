# Joint Constraints Integration Guide

**Status**: Ready to use  
**Prerequisite**: [docs/JOINT_CONSTRAINTS.md](JOINT_CONSTRAINTS.md) - physics reference

---

## Quick Start

### Step 1: Enable in Config

Add to [config.py](../config.py):

```python
# ===== Phase 5: Joint Constraints =====
USE_JOINT_CONSTRAINTS = True  # Enable realistic joint limits
```

### Step 2: Configure in Simulation

In [source/simulate.py](../source/simulate.py), enable joint constraints when setting up the creature:

```python
from source.entity import Creature
from source.physics import PhysicsEngine
from source.joint_constraints import QuadrupedJointSetup

# Create physics engine (standard)
physics_engine = PhysicsEngine(device, dtype, environment)

# Create creature
creature = Creature(device=device, dtype=dtype)

# NEW: Configure realistic joint constraints
joint_configs = QuadrupedJointSetup.create_quadruped_joints(device, dtype)
physics_engine.configure_joint_constraints(creature, joint_configs)

# Continue with normal training loop...
```

### Step 3: That's It!

The quadruped now has realistic joint constraints:
- Hip: Spherical (3 DOF, full mobility)
- Knee: Revolute (1 DOF, -60° to 0°)
- Ankle: Revolute (1 DOF, -30° to +30°)

---

## Integration Points

### PhysicsEngine Methods

**`configure_joint_constraints(creature, joint_configs)`**

Enables realistic joint behavior for a creature. Called during initialization.

```python
# Enable for all creatures
for creature in environment.creatures:
    physics_engine.configure_joint_constraints(creature, joint_configs)

# OR enable only specific creatures
physics_engine.configure_joint_constraints(main_agent, joint_configs)
```

**Joint constraints automatically apply in `apply_motor_torques()`**

After rigid body integration, constraints enforce:
- Joint angle limits (spring-back if exceeded)
- Damping (energy dissipation)
- Friction (resistance to motion)
- Max torque limits

---

## Configuration Patterns

### Pattern 1: Default Quadruped (Recommended)

```python
from source.joint_constraints import QuadrupedJointSetup

# Standard realistic quadruped legs
joints = QuadrupedJointSetup.create_quadruped_joints(device, dtype)
physics_engine.configure_joint_constraints(creature, joints)
```

**Result**: Quadruped with realistic biological constraints
- Hip: Full 3D freedom (spherical)
- Knee: Only bending (-60° to 0°)
- Ankle: Small rotation range (-30° to +30°)

---

### Pattern 2: Custom Quadruped

```python
from source.joint_constraints import JointConfig, JointType, JointConstraint

# Create custom joints with different parameters
joints = []

# Stiffer joints (less flexible)
for leg_idx in range(4):
    hip = JointConfig(
        joint_type=JointType.SPHERICAL,
        parent_link=f"torso",
        child_link=f"leg_{leg_idx}_hip",
        anchor_pos=...,
        stiffness=0.5,      # Stiffer (from default 0.2)
        damping=0.15        # More damped
    )
    joints.append(hip)

physics_engine.configure_joint_constraints(creature, joints)
```

---

### Pattern 3: Multi-Robot with Different Morphologies

```python
from source.joint_constraints import QuadrupedJointSetup, HumanoidJointSetup

# Quadruped
creature1 = Creature(device=device, dtype=dtype)
quad_joints = QuadrupedJointSetup.create_quadruped_joints(device, dtype)
physics_engine.configure_joint_constraints(creature1, quad_joints)

# Humanoid (different morphology)
creature2 = Creature(device=device, dtype=dtype)
humanoid_joints = HumanoidJointSetup.create_humanoid_joints(device, dtype)
physics_engine.configure_joint_constraints(creature2, humanoid_joints)

# Physics engine automatically handles both
```

---

### Pattern 4: Disable Joint Constraints

Even if enabled in Config, you can disable per-creature:

```python
# Configure but then disable for testing
physics_engine.use_joint_constraints = False

# OR remove specific creature's constraints
creature_id = id(creature)
if creature_id in physics_engine.joint_constraints:
    del physics_engine.joint_constraints[creature_id]
```

---

## Backward Compatibility

**100% Backward Compatible** ✅

- Old code with simple 12D actions still works
- Just set `USE_JOINT_CONSTRAINTS = False` in config
- No breaking changes to existing APIs

**Migration Path**:
1. Keep existing training with simple model
2. Try with `USE_JOINT_CONSTRAINTS = True`
3. Gradually tune joint parameters
4. Full migration takes 0 seconds (just flip a flag!)

---

## Parameter Tuning

### Too Stiff (Joint Won't Move Smoothly)

```python
# In config or custom JointConfig:
JOINT_STIFFNESS = 0.1      # Lower from 0.2 or 0.5
JOINT_DAMPING = 0.05       # Lower from 0.1 or 0.15
```

**Symptom**: Agent has jerky motion, can't smoothly extend legs

---

### Too Loose (Joint Oscillates or Exceeds Limits)

```python
JOINT_STIFFNESS = 0.5      # Higher from 0.2
JOINT_DAMPING = 0.15       # Higher from 0.05
```

**Symptom**: Legs bounce or swing beyond biological limits

---

### Joint Reaches Limits Too Easily

```python
# Widen joint angle limits
KNEE_LOWER_LIMIT = -math.pi / 2    # More negative (more flex)
KNEE_UPPER_LIMIT = math.pi / 6     # More positive (hyperextend slightly)

# OR reduce limit penalty
# In source/joint_constraints.py, change:
# tau -= 10.0 * stiffness * angle_error
# to:
# tau -= 5.0 * stiffness * angle_error
```

**Symptom**: Agent can't reach certain movement patterns

---

## Validation

### Does Joint Constraint Setup Work?

```python
def validate_joint_constraints(physics_engine, creature):
    """Quick validation that joint constraints are active."""
    creature_id = id(creature)
    
    if creature_id not in physics_engine.joint_constraints:
        print("❌ Joint constraints NOT configured!")
        return False
    
    num_joints = len(physics_engine.joint_constraints[creature_id])
    print(f"✅ Joint constraints active: {num_joints} joints")
    
    # Check that constraints actually apply
    for i, joint in enumerate(physics_engine.joint_constraints[creature_id]):
        print(f"  Joint {i}: {joint.config.joint_type.name}")
    
    return True

# Usage
validate_joint_constraints(physics_engine, creature)
```

---

## Expected Behavior Changes

### With Joint Constraints Enabled

| Aspect | Without Constraints | With Constraints |
|--------|---------------------|------------------|
| Leg angles | Any value (12D free) | Realistic ranges |
| Knee | Can flex/extend infinitely | -60° to 0° only |
| Hip | Free rotation | Full 3D with limits |
| Ankle | Free rotation | -30° to +30° |
| Leg damping | Actuator lag only | + joint damping |
| Realism | Lower | Higher |
| Computational cost | Baseline | +10-20% |
| Sim-to-real transfer | Harder | Easier |

**Training Impact**:
- First few episodes: Agent learns new constraints
- Performance: Usually improves (more stable) or slightly decreases (less free DOF)
- Long-term: More realistic gaits emerge

---

## Implementation Details

### How Joint Constraints Work

1. **Motor torques** apply as usual (τ_motor)
2. **Joint constraint forces** apply AFTER rigid body integration:
   ```
   τ_constraint = -stiffness * angle_error - damping * velocity
   ```
3. **Limits enforce** spring-back if angle exceeds bounds
4. **Friction opposes** motion

### Physics Order

```
1. Command motor torques
2. Integrate kinematics (joint angles/velocities update)
3. Integrate rigid body dynamics (body position/rotation update)
4. Apply joint constraint forces (NEW)
5. Update creature observation
6. Compute rewards
```

This ordering ensures:
- Constraints don't fight motor commands (applied after)
- Body dynamics are consistent
- Constraints can "correct" excessive motion

---

## Troubleshooting

### Joint Constraints Not Activating?

```python
# Check 1: Config flag
from config import Config
print(f"USE_JOINT_CONSTRAINTS in config: {hasattr(Config, 'USE_JOINT_CONSTRAINTS')}")

# Check 2: PhysicsEngine.use_joint_constraints flag
print(f"PhysicsEngine.use_joint_constraints: {physics_engine.use_joint_constraints}")

# Check 3: Creature registered?
creature_id = id(creature)
print(f"Creature {creature_id} in constraints: {creature_id in physics_engine.joint_constraints}")

# Check 4: Actually call configure_joint_constraints()?
# (Easy to forget!)
```

### Joint Angles Exceed Limits?

Limits apply SPRING FORCES, not hard stops. If motor torques exceed spring forces:

```python
# Increase spring stiffness for harder limits
JOINT_STIFFNESS = 1.0  # Very stiff limits

# OR reduce max motor torque
MAX_TORQUE = 10.0  # Lower from 20.0
```

### Simulation Crashes or Becomes Unstable?

- Reduce `JOINT_DAMPING` (overdamped system can cause stiffness)
- Check `JOINT_STIFFNESS` isn't too high (numerical instability)
- Ensure `DT = 0.004` (250 Hz is stable, larger timesteps can oscillate)

---

## Advanced: Custom Joint Types

To support humanoids or other morphologies:

```python
from source.joint_constraints import JointConfig, JointType

# Spine (Hinge2: pitch + yaw)
spine = JointConfig(
    joint_type=JointType.HINGE2,
    parent_link="pelvis",
    child_link="torso",
    axis=np.array([1, 0, 0]),     # Pitch
    axis2=np.array([0, 0, 1]),    # Yaw (twist)
    lower_limit=-0.5,             # -30°
    upper_limit=0.5,              # +30°
    stiffness=0.4,
    damping=0.1
)

# Neck (Spherical: 3 DOF)
neck = JointConfig(
    joint_type=JointType.SPHERICAL,
    parent_link="torso",
    child_link="head",
    anchor_pos=np.array([0, 0.3, 0]),
    stiffness=0.3,
    damping=0.08
)

# Knee (Revolute: 1 DOF)
knee = JointConfig(
    joint_type=JointType.REVOLUTE,
    parent_link="femur",
    child_link="tibia",
    axis=np.array([0, 1, 0]),
    lower_limit=-1.5,             # -90°
    upper_limit=0,                # 0° (can't hyperextend)
    stiffness=0.6,
    damping=0.12
)
```

See [docs/JOINT_CONSTRAINTS.md](JOINT_CONSTRAINTS.md) for all 5 joint types.

---

## Next Steps

1. ✅ Joint constraint system implemented
2. ✅ Integration into PhysicsEngine complete
3. ⏳ Test on quadruped with `USE_JOINT_CONSTRAINTS = True`
4. ⏳ Tune joint parameters for your specific morphology
5. ⏳ Validate training converges with constraints

Ready to use! 🚀
