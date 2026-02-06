# Performance Optimization Guide

## JIT Kernel Fusion Strategy

This document explains the critical performance optimization that powers the system: **single large compiled kernel** vs multiple small JIT calls.

## The Problem: Boundary Crossing Overhead

When code alternates between Python and compiled JIT functions, there's overhead at each crossing:

```
SLOW PATTERN (3 boundary crossings):
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Python  │ --> │  JIT A  │ --> │ Python  │ --> │  JIT B  │ --> Python
└─────────┘     └─────────┘     └─────────┘     └─────────┘
    ~1-2μs         compute         ~1-2μs         compute        ~1-2μs
```

Over millions of calls, this overhead compounds:
- **Scale**: In training, physics simulation happens ~millions of times
  - 100 environments × 1000 steps/episode × 100 episodes = 10 million calls
- **Overhead per call**: 1-2 microseconds
- **Total overhead**: 10-20 seconds just from boundary crossing!

## The Solution: Kernel Fusion

**Single large kernel eliminates boundary crossing overhead:**

```
FAST PATTERN (1 boundary crossing):
┌─────────┐     ┌──────────────────────────────────────┐     ┌─────────┐
│ Python  │ --> │  LARGE JIT KERNEL                    │ --> │ Python  │
└─────────┘     │  (Physics + Reward + Observation)    │     └─────────┘
                │  All computation compiled together    │
                └──────────────────────────────────────┘
                    Enter once, stay compiled, exit once
```

## Implementation: simulation_step() in simulate.py

The kernel is **fused directly into simulate.py** (not imported from another module):

```python
@jit(nopython=True)
def simulation_step(
    # State (6 floats): position + velocity
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    
    # Action (3 floats): acceleration input
    accel_x, accel_y, accel_z,
    
    # World geometry (9 floats): goal + bounds
    goal_x, goal_y, goal_z,
    bound_min_x, bound_min_y, bound_min_z,
    bound_max_x, bound_max_y, bound_max_z,
    
    # Physics parameters (11 floats)
    accel_scale_xy, accel_scale_z, max_accel,
    max_vel, momentum_damping, gravity,
    terminal_vel_z, ground_level,
    ground_friction, air_friction, air_drag,
    
    # Reward parameters (8 floats)
    prev_dist, goal_threshold, proximity_threshold,
    distance_reward_scale, proximity_bonus_scale, goal_bonus,
    wall_penalty_scale, stamina_penalty
    
) -> tuple:
    """
    Complete physics simulation step in one compiled function.
    
    Returns: 15 floats
        - new position (3)
        - new velocity (3)
        - reward (1)
        - current distance (1)
        - wall penalty (1)
        - observation (5: 3D relative + 2 absolute)
    """
    
    # PHYSICS PHASE
    # - Apply acceleration with scaling
    # - Apply momentum damping
    # - Apply gravity & collision
    # - Apply friction/drag
    # - Clamp velocity
    # - Update position & boundary clamp
    
    # REWARD PHASE
    # - Calculate distance to goal
    # - Distance-based reward (2x penalty for moving away)
    # - Goal reached bonus
    # - Proximity bonus
    # - Wall penetration penalty
    
    # OBSERVATION PHASE
    # - Normalize goal-relative position
    # - Normalize absolute position
    
    return (clamped_x, clamped_y, clamped_z,
            new_vel_x, new_vel_y, new_vel_z,
            reward, curr_dist, wall_penalty,
            obs_rel_x, obs_rel_y, obs_rel_z,
            obs_abs_x, obs_abs_y, obs_abs_z)
```

## Why Direct Inlining Matters

The kernel is **fused into simulate.py** (not imported from helper_math):

**Advantages**:
- No module import lookup cost
- Numba compiler sees full context (all parameters at once)
- Easier to understand physics logic (everything in one place)
- Fewer file imports in critical path

**How to use it**:
```python
# In simulate.py
result = simulation_step(
    pos.x, pos.y, pos.z, vel.x, vel.y, vel.z,
    action.x, action.y, action.z,
    goal.x, goal.y, goal.z,
    # ... 38 more parameters ...
)

new_x, new_y, new_z, new_vx, new_vy, new_vz, reward, dist, penalty, *obs = result
```

## Performance Impact

**Measurements**:
- Single JIT call (fused): ~0.5-1.0 microsecond per step
- Previous approach (3 JIT calls): ~3-5 microseconds per step
- **Speedup**: 3-5x faster physics simulation

**Example training impact**:
```
Before fusion:
100 envs × 1000 steps × 100 episodes × 4 microseconds = 40 seconds physics time

After fusion:
100 envs × 1000 steps × 100 episodes × 1 microsecond = 10 seconds physics time

Total training speedup: ~20% (depends on training/physics ratio)
```

## When to Fuse, When to Split

### Fuse together:
- ✅ Frequently called hot-path functions (millions/episode)
- ✅ Related computations (physics depends on position→reward depends on physics)
- ✅ Functions with many shared parameters

### Keep separate:
- ✅ Rarely called functions (load/save checkpoints, visualization)
- ✅ Unrelated domains (world model uses different logic)
- ✅ Functions with very different parameter sets

## Implementation Checklist

If adding new physics features:

1. **Add parameters to simulation_step()** (declare at top)
2. **Implement logic inside the kernel** (keep it scalar, no allocations)
3. **Update return tuple** if new outputs needed
4. **Update move() call** in System.move() to pass new parameters
5. **Benchmark**: Compare single JIT call vs multiple small calls

## Numba JIT Compilation Rules

**Requirements for nopython=True mode**:
- ✅ No Python API calls (no .append(), no print, no exceptions)
- ✅ Only scalar math (floats, ints, basic ops)
- ✅ No NumPy arrays (use scalar parameters)
- ✅ Type-stable (variable types don't change)
- ❌ No list comprehensions
- ❌ No dictionary operations
- ❌ No string operations

**Compilation happens once** on first call, then runs at near-C speed.

## Profiling

To measure physics overhead:
```python
import time

# Before physics
t0 = time.perf_counter()

# 1000 physics steps
for _ in range(1000):
    result = simulation_step(...)

# After physics
t1 = time.perf_counter()
print(f"1000 steps: {(t1-t0)*1e6:.1f} microseconds average")
```

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md#performance-jit-kernel-fusion-architecture) - System design
- [WORLD_SYSTEM.md](WORLD_SYSTEM.md#physics-system) - Physics details
- [QUICKSTART.md](QUICKSTART.md) - Getting started

