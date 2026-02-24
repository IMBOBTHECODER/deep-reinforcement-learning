# Performance Optimization Guide

This document covers all current optimizations in the quadruped training system.
Arranged from innermost hot-path to outer training loop.

---

## 1. Numba `@njit` for Scalar Physics (CPU)

Three functions in `physics.py` are decorated with `@njit(cache=True)`.
They handle the parts of the per-creature physics step that are inherently
scalar — quaternion math and the rigid-body integrator — where PyTorch has
too much per-call overhead for only 4–12 numbers.

### `_jit_quat_to_rot(w, x, y, z)` → (3, 3)
Builds a rotation matrix from a quaternion using 9 scalar assignments.
Called every physics step per creature.  Without `@njit` the cost is a
Python `np.array([[...]])` call with list-of-list allocation overhead.

### `_jit_quat_to_euler(w, x, y, z)` → (pitch, yaw, roll)
Six trig calls and three `atan2`/`asin` branches compiled to native code.
Used to sync the creature's orientation tensor from its `RigidBody`.

### `_jit_integrate_body(pos, vel, ω, q, F, τ, m, I_diag, dt, ω_max)` → (qw, qx, qy, qz)
The full semi-implicit Euler integrator:
1. Linear dynamics — velocity + position update
2. World-frame inertia: `I_world = R @ diag(I_diag) @ R.T`
3. Euler equations: `I α = τ − ω × (Iω)`, solved with `np.linalg.solve`
4. Angular velocity clamp
5. Quaternion integration + normalisation

`pos`, `linear_vel`, `angular_vel` are updated **in-place** (NumPy shared
memory).  Only the new quaternion scalars are returned.

```python
# physics.py — called N times per step (once per creature)
new_qw, new_qx, new_qy, new_qz = _jit_integrate_body(
    body.pos, body.linear_vel, body.angular_vel,
    body.orientation.w, body.orientation.x,
    body.orientation.y, body.orientation.z,
    force_accum, torque_accum,
    mass, np.diag(inertia_tensor),
    dt, MAX_ANGULAR_VELOCITY,
)
```

All three functions are **cache=True** — compiled once, persisted to
`__pycache__`, zero recompile cost on subsequent runs.

**Fallback**: If `numba` is not installed the module defines a no-op `njit`
decorator; the functions run as plain NumPy with identical results.

---

## 2. Vectorised Contact Detection (NumPy, per-creature path)

Inside `_update_joint_dynamics_cpu`, the old 4-iteration Python `for` loop
over feet is replaced with NumPy slice operations on `(4,)` arrays:

```python
# One .numpy() call instead of four float() casts
foot_z_np    = foot_positions[:, 2].detach().cpu().numpy()   # (4,)
contact_mask = foot_z_np <= ground_level + threshold          # (4,) bool
pens         = np.maximum(0, ground_level - foot_z_np) * contact_mask
spring       = contact_stiffness * pens
resti        = contact_restitution * (-bvz) * spring / np.maximum(pens, 0.001)
contact_force_total[2] = np.sum(np.maximum(0, spring - damper + resti) * contact_mask)
```

The `* contact_mask` at the end prevents spurious forces on airborne feet
when `body_vz < 0` (falling body makes damper term negative).

---

## 3. Batched PyTorch CUDA Ops (`_step_batch_gpu`)

When `VECTORIZED_PHYSICS = True` and a CUDA device is present, all N
environments are processed together instead of in a Python `for` loop.

| Section | Tensors | What replaces |
|---|---|---|
| Joint dynamics | `(N, 12)` elementwise | N × per-creature calls |
| Forward kinematics | 4 × `torch.bmm` on `(N, 3, 3)` | N × 4-leg FK loops |
| Contact detection | `(N, 4)` comparison | N × 4-foot Python loops |
| Spring-damper | `(N, 4)` elementwise | N × 4-foot force loops |

PyTorch dispatches each op to cuBLAS / cuDNN and selects the optimal
grid/block layout automatically — no Numba CUDA kernels needed.  This
eliminates the old "Grid size 1" under-utilisation warning that appeared
when a Numba kernel was launched with only 12 work items.

```
Before (_step_batch_gpu ran a Python for-loop):
  16 envs × 4 legs × 3 segments  =  192 separate function calls
  16 × 4 FK calls inside Python loop

After (batched):
  Joint dynamics:  3 elementwise ops on (16, 12)       ← 1 CUDA launch
  FK:              4 × bmm on (16, 3, 3)               ← 4 CUDA launches
  Contact/forces:  5 elementwise ops on (16, 4)        ← 1 CUDA launch
```

### `_batch_forward_kinematics` detail

```python
# Rotation matrices for all N creatures — one stack call per axis
Rx = torch.stack([...], dim=1)   # (N, 3, 3)
Ry = torch.stack([...], dim=1)
Rz = torch.stack([...], dim=1)
R  = torch.bmm(torch.bmm(Rz, Ry), Rx)   # (N, 3, 3) combined

# Each of the 4 legs: cumulative pitch-plane offset + hip rotation
for leg_idx in range(4):          # only 4 iterations (not N)
    p3_world = torch.bmm(R, p3.unsqueeze(-1)).squeeze(-1)   # (N, 3)
    feet.append(hip_world + p3_world)

return torch.stack(feet, dim=1)  # (N, 4, 3)
```

---

## 4. Batch Encoder in PPO (`forward_sequence`)

```python
# Before — T × num_envs × PPO_EPOCHS separate (1, 34) calls:
for obs in obs_list:                    # 1 024 iterations
    (mu, log_std), value, state = model(obs, ...)

# After — one (T, 34) matmul, then LSTM loop for done-masking:
obs_seq = torch.cat(obs_list, dim=0)    # (T, 34)
(mu_seq, log_std_seq), val_seq, _ = model.forward_sequence(obs_seq, dones=dones)
```

| | Before | After |
|---|---|---|
| Encoder calls per episode | T × envs × epochs = 65 536 | envs × epochs = 64 |
| Shape per call | (1, 34) | (1 024, 34) |

---

## 5. ResourceMonitor — Background Thread

The `ResourceMonitor` class in `simulate.py` samples CPU / RAM / GPU / VRAM
in a **daemon thread** — it never touches the training hot-path.

```
Training thread (hot path):
  step 0 → step 1 → step 2 → ...   ← never blocks

Monitor thread (background, every 5 s):
  sample → log → print \r status bar
```

**Live console bar** (overwrites the same line):
```
  CPU [████████░░]  82% RAM   1234/16384MB  | GPU[Tesla T4] [██████░░░░]  61% VRAM   4096/16160MB  | ep=3 step=412 r=+0.87
```

**Every sample is written to `training.log`**:
```
2026-02-24 10:31:05 - INFO - [RESOURCES] ep=   3 step=   412 reward=+0.870 | CPU= 82.1%  RAM=  1234.5MB | GPU= 61.0%  VRAM=  4096.0/16160MB (25.3%)
```

| Parameter | Default | Description |
|---|---|---|
| `interval` | 5.0 s | How often to sample |
| GPU backend | pynvml | Accurate direct NVML query |
| Fallback | GPUtil | If pynvml not installed |

---

## 6. Numba JIT Rules

`@njit` functions must follow these rules (enforced at compile time):

| Allowed | Not allowed |
|---|---|
| Scalar math (float, int) | Python API calls (print, append) |
| NumPy array creation (`np.empty`, `np.zeros`) | List comprehensions |
| `math.*` trig | Dictionary operations |
| `np.linalg.solve` / `np.linalg.norm` | String operations |
| In-place array ops (`arr[i] += x`) | Object creation / class methods |

Compilation happens **once** on the first call, then runs at near-C speed.
`cache=True` persists the compiled binary to `__pycache__` so subsequent
program starts skip recompilation entirely.

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and data flow
- [PHYSICS.md](PHYSICS.md) — Full physics engine reference
- [PHYSICS_ENGINE_UPGRADES.md](PHYSICS_ENGINE_UPGRADES.md) — 250 Hz, contacts, vectorization

