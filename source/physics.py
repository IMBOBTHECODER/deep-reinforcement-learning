"""
Advanced Physics Engine for Quadruped Robot.
See docs/PHYSICS.md for detailed feature documentation.

Key Classes: Quaternion (gimbal-lock-free), RigidBody (full dynamics).
GPU Acceleration: batched PyTorch CUDA ops in _step_batch_gpu (no Numba kernels –
PyTorch selects optimal grid/block layout automatically).
"""

import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os

# GPU acceleration – detected once at import; physics uses PyTorch CUDA ops
# (no Numba CUDA kernels: PyTorch handles grid/block sizing automatically,
#  which avoids "Grid size 1" under-utilisation warnings on small batches)
HAS_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Numba CPU JIT – tight scalar loops (quaternion math + rigid-body integrator).
# Falls back gracefully to plain NumPy when numba is not installed.
# NOTE: no CUDA kernels here – GPU work is done by PyTorch in _step_batch_gpu.
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # noqa: E306
        """No-op replacement when numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


@njit(cache=True)
def _jit_quat_to_rot(w, x, y, z):
    """
    Quaternion (w, x, y, z) → 3×3 rotation matrix (float64 NumPy array).
    @njit eliminates per-call Python overhead – ~5× faster than np.array() for
    scalar quaternions called N times per physics step.
    """
    R = np.empty((3, 3))
    R[0, 0] = 1.0 - 2.0 * (y*y + z*z);  R[0, 1] = 2.0 * (x*y - w*z);  R[0, 2] = 2.0 * (x*z + w*y)
    R[1, 0] = 2.0 * (x*y + w*z);  R[1, 1] = 1.0 - 2.0 * (x*x + z*z);  R[1, 2] = 2.0 * (y*z - w*x)
    R[2, 0] = 2.0 * (x*z - w*y);  R[2, 1] = 2.0 * (y*z + w*x);  R[2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return R


@njit(cache=True)
def _jit_quat_to_euler(w, x, y, z):
    """
    Quaternion (w, x, y, z) → (pitch, yaw, roll) in radians.
    @njit removes repeated Python-level trig for every body per step.
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if sinp < -1.0:
        sinp = -1.0
    elif sinp > 1.0:
        sinp = 1.0
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return pitch, yaw, roll


@njit(cache=True)
def _jit_integrate_body(pos, linear_vel, angular_vel,
                        qw, qx, qy, qz,
                        force_accum, torque_accum,
                        mass, inertia_diag,
                        dt, max_angular_vel):
    """
    Semi-implicit Euler rigid-body integrator.  @njit removes the Python-object
    overhead that accumulates when this is called N times (once per creature) in
    the sequential loop inside _step_batch_gpu and _update_joint_dynamics_cpu.

    pos / linear_vel / angular_vel are modified in-place (they share memory with
    the caller's NumPy arrays).  The updated quaternion is returned as scalars.

    Args:
        inertia_diag:   (3,) principal moments of inertia (local-frame diagonal).
        force_accum:    (3,) accumulated forces; gravity must be baked in before call.
        max_angular_vel: rad/s clamp applied to angular velocity.
    Returns:
        (qw, qx, qy, qz) – normalised updated quaternion.
    """
    # ── Linear dynamics ───────────────────────────────────────────────────────
    inv_m = 1.0 / mass
    linear_vel[0] += force_accum[0] * inv_m * dt
    linear_vel[1] += force_accum[1] * inv_m * dt
    linear_vel[2] += force_accum[2] * inv_m * dt
    pos[0] += linear_vel[0] * dt
    pos[1] += linear_vel[1] * dt
    pos[2] += linear_vel[2] * dt

    # ── World-frame inertia: I_world = R @ diag(d) @ R.T ─────────────────────
    # I_world[i,j] = sum_k  R[i,k] * d[k] * R[j,k]
    R = _jit_quat_to_rot(qw, qx, qy, qz)
    I_world = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += R[i, k] * inertia_diag[k] * R[j, k]
            I_world[i, j] = s

    # ── Euler equations: I α = τ − ω × (I ω) ─────────────────────────────────
    Iw0 = I_world[0,0]*angular_vel[0] + I_world[0,1]*angular_vel[1] + I_world[0,2]*angular_vel[2]
    Iw1 = I_world[1,0]*angular_vel[0] + I_world[1,1]*angular_vel[1] + I_world[1,2]*angular_vel[2]
    Iw2 = I_world[2,0]*angular_vel[0] + I_world[2,1]*angular_vel[1] + I_world[2,2]*angular_vel[2]

    gyro0 = angular_vel[1] * Iw2 - angular_vel[2] * Iw1
    gyro1 = angular_vel[2] * Iw0 - angular_vel[0] * Iw2
    gyro2 = angular_vel[0] * Iw1 - angular_vel[1] * Iw0

    net_tau = np.empty(3)
    net_tau[0] = torque_accum[0] - gyro0
    net_tau[1] = torque_accum[1] - gyro1
    net_tau[2] = torque_accum[2] - gyro2

    angular_accel = np.linalg.solve(I_world, net_tau)
    angular_vel[0] += angular_accel[0] * dt
    angular_vel[1] += angular_accel[1] * dt
    angular_vel[2] += angular_accel[2] * dt

    # Clamp angular velocity
    ang_mag = math.sqrt(angular_vel[0]**2 + angular_vel[1]**2 + angular_vel[2]**2)
    if ang_mag > max_angular_vel:
        s = max_angular_vel / ang_mag
        angular_vel[0] *= s;  angular_vel[1] *= s;  angular_vel[2] *= s

    # ── Quaternion integration: q += 0.5 * dt * q ⊗ ω ───────────────────────
    half_dt = 0.5 * dt
    dqw = half_dt * (-qx * angular_vel[0] - qy * angular_vel[1] - qz * angular_vel[2])
    dqx = half_dt * ( qw * angular_vel[0] + qy * angular_vel[2] - qz * angular_vel[1])
    dqy = half_dt * ( qw * angular_vel[1] - qx * angular_vel[2] + qz * angular_vel[0])
    dqz = half_dt * ( qw * angular_vel[2] + qx * angular_vel[1] - qy * angular_vel[0])
    qw += dqw;  qx += dqx;  qy += dqy;  qz += dqz

    mag = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if mag > 1e-6:
        qw /= mag;  qx /= mag;  qy /= mag;  qz /= mag

    return qw, qx, qy, qz


# Multi-threading pool for CPU physics fallback
def _get_optimal_thread_count():
    """Auto-detect optimal number of worker threads."""
    try:
        num_cpus = os.cpu_count() or 4
        # Use all CPUs except 1 for OS/other tasks (or at most 8 for practical limits)
        return max(2, min(num_cpus - 1, 8))
    except:
        return 4


@dataclass
class Quaternion:
    """Represents orientation as quaternion (w, x, y, z) - no gimbal lock."""
    w: float
    x: float
    y: float
    z: float
    
    @classmethod
    def identity(cls):
        """Identity quaternion (no rotation)."""
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float):
        """
        Convert Euler angles (pitch, yaw, roll) to quaternion.
        Order: applied as Rz(yaw) * Ry(pitch) * Rx(roll)
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    def to_euler(self) -> Tuple[float, float, float]:
        """Convert quaternion back to Euler angles (pitch, yaw, roll) via @njit helper."""
        return _jit_quat_to_euler(self.w, self.x, self.y, self.z)
    
    def normalize(self):
        """Normalize quaternion to unit length."""
        mag = math.sqrt(self.w * self.w + self.x * self.x + 
                        self.y * self.y + self.z * self.z)
        if mag > 1e-6:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    def __mul__(self, other):
        """Quaternion multiplication (non-commutative)."""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        raise TypeError("Can only multiply Quaternion with Quaternion")
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3×3 rotation matrix (delegates to @njit helper)."""
        return _jit_quat_to_rot(self.w, self.x, self.y, self.z)


@dataclass
class RigidBody:
    """Rigid body with full dynamics: position, orientation, linear/angular velocity.
    See docs/PHYSICS.md for detailed equations.
    """
    pos: np.ndarray              # (3,) COM position
    orientation: Quaternion      # Rotation quaternion
    linear_vel: np.ndarray       # (3,) COM velocity
    angular_vel: np.ndarray      # (3,) Angular velocity (rad/s)
    
    mass: float = 1.0
    inertia_tensor: np.ndarray = None  # (3, 3) in local frame
    
    force_accum: np.ndarray = None
    torque_accum: np.ndarray = None
    
    def __post_init__(self):
        if self.inertia_tensor is None:
            self.inertia_tensor = np.eye(3) * (0.4 * self.mass * 0.25)
        
        if self.force_accum is None:
            self.force_accum = np.zeros(3)
        
        if self.torque_accum is None:
            self.torque_accum = np.zeros(3)
    
    def add_force(self, force: np.ndarray, point: Optional[np.ndarray] = None):
        """Add force at COM (or at point, creating torque)."""
        self.force_accum += force
        if point is not None:
            self.torque_accum += np.cross(point, force)
    
    def clear_forces(self):
        """Clear accumulated forces and torques."""
        self.force_accum.fill(0)
        self.torque_accum.fill(0)
    
    def integrate(self, dt: float, gravity: float = 9.8):
        """
        Integrate rigid body dynamics (semi-implicit Euler + Euler equations).

        Dispatches to @njit _jit_integrate_body which eliminates Python-object
        overhead for the N-creature sequential loop in _step_batch_gpu.

        pos / linear_vel / angular_vel are updated in-place by the JIT function.
        The legacy ``gravity`` parameter (default 9.8) is baked into force_accum
        before the call; all hot-path callers pass gravity=0 because gravity is
        already pre-accumulated via add_force().
        """
        from config import Config

        # Bake legacy gravity arg into a temporary copy of force_accum.
        # Hot-path: gravity=0 → fa == self.force_accum (no copy).
        fa = self.force_accum
        if gravity != 0.0:
            fa = self.force_accum.copy()
            fa[2] -= gravity * self.mass

        new_qw, new_qx, new_qy, new_qz = _jit_integrate_body(
            self.pos, self.linear_vel, self.angular_vel,
            self.orientation.w, self.orientation.x,
            self.orientation.y, self.orientation.z,
            fa, self.torque_accum,
            self.mass, np.diag(self.inertia_tensor),
            dt, float(Config.MAX_ANGULAR_VELOCITY),
        )
        # pos / linear_vel / angular_vel already updated in-place above.
        self.orientation.w = new_qw
        self.orientation.x = new_qx
        self.orientation.y = new_qy
        self.orientation.z = new_qz
        self.clear_forces()


class PhysicsEngine:
    """
    Advanced physics engine with rigid body dynamics, quaternion orientation, and realistic contacts.
    
    DESIGN: Physics is the PRIMARY simulation (realism-first). DRL is a PLUGIN on top.
    
    Features: 
    - Gravity integration, spring-damper contacts, joint velocity clamping, agent-centered world
    - Quaternion-based orientation (no gimbal lock), inertia tensor, Euler equations
    - GPU Acceleration: batched PyTorch CUDA ops via step_batch() (PyTorch handles grid/block sizing)
    - Phase 1: Actuator response lag (first-order lag model)
    - Phase 2: Improved friction model (Coulomb + viscous damping)
    - Phase 4: Energy tracking (mechanical power consumption)
    
    Backward Compatible: All realism features can be disabled via Config parameters.
    """
    
    def __init__(self, device, dtype, environment):
        from config import Config
        
        self.device = device
        self.dtype = dtype
        self.env = environment
        
        # Physics constants
        self.gravity = Config.GRAVITY
        self.dt = Config.DT  # 100 Hz
        
        # Quadruped leg dynamics
        self.joint_damping = Config.JOINT_DAMPING
        self.max_torque = Config.MAX_TORQUE
        self.segment_length = Config.SEGMENT_LENGTH
        self.max_joint_velocity = Config.MAX_JOINT_VELOCITY
        
        # ===== Phase 1: Actuator Response Lag =====
        self.actuator_response_time = getattr(Config, 'ACTUATOR_RESPONSE_TIME', 0.0)
        # Per-joint actuator state (tracks applied torque for lag simulation)
        # Will be initialized per creature in apply_motor_torques()
        
        # Contact properties
        self.ground_level = Config.GROUND_LEVEL
        self.foot_height_threshold = Config.FOOT_HEIGHT_THRESHOLD
        self.contact_stiffness = Config.CONTACT_STIFFNESS
        self.contact_damping = Config.CONTACT_DAMPING
        self.contact_restitution = Config.CONTACT_RESTITUTION  # Coefficient of restitution for bouncing
        
        # ===== Phase 2: Improved Friction Model =====
        self.friction_model = getattr(Config, 'FRICTION_MODEL', 'coulomb+viscous')
        self.friction_coeff_static = getattr(Config, 'FRICTION_COEFFICIENT_STATIC', 0.9)
        self.friction_coeff_kinetic = getattr(Config, 'FRICTION_COEFFICIENT_KINETIC', 0.85)
        self.friction_viscous_damping = getattr(Config, 'FRICTION_VISCOUS_DAMPING', 0.05)
        self.friction_slip_threshold = getattr(Config, 'FRICTION_SLIP_VELOCITY_THRESHOLD', 0.01)
        
        # ===== Phase 3: Friction Cones (Directional Constraint) =====
        self.use_friction_cones = getattr(Config, 'USE_FRICTION_CONES', True)
        self.friction_cone_damping = getattr(Config, 'FRICTION_CONE_DAMPING', 0.3)
        
        # Legacy support (simple model)
        if not hasattr(Config, 'FRICTION_MODEL'):
            self.friction_coeff = Config.GROUND_FRICTION_COEFFICIENT
        
        # ===== Phase 4: Energy Tracking =====
        self.track_energy = getattr(Config, 'TRACK_ENERGY_CONSUMPTION', False)
        self.motor_efficiency = getattr(Config, 'MOTOR_EFFICIENCY', 0.80)
        if not (0.0 < self.motor_efficiency <= 1.0):
            self.motor_efficiency = 0.80
        
        # Reward parameters
        self.com_distance_threshold = Config.COM_DISTANCE_THRESHOLD
        self.contact_reward = Config.CONTACT_REWARD
        self.energy_penalty = Config.ENERGY_PENALTY
        self.tilt_penalty = Config.TILT_PENALTY
        
        self.max_pitch_roll = Config.MAX_PITCH_ROLL
        
        # Multi-threading setup for CPU fallback
        num_threads = Config.NUM_PHYSICS_THREADS or _get_optimal_thread_count()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="physics_"
        ) if not HAS_CUDA else None
        
        # Per-creature rigid bodies.
        # Previously a single self.body was shared across all parallel environments,
        # causing two bugs:
        #   1. Race condition: threads in ThreadPoolExecutor overwrote each other's
        #      body state, serialising what should be parallel work.
        #   2. Wrong physics: env A's creature.pos was overwritten by env B's body.
        # Bodies are now created lazily per creature in _get_or_create_body().
        self._bodies = {}

        self.agent_local_pos = torch.zeros(3, device=device, dtype=dtype)
        self.atanh_eps = Config.ATANH_EPSILON
        self.log_eps = Config.LOG_EPSILON

        # ===== JOINT CONSTRAINTS (Optional, Phase 5) =====
        self.joint_constraints = {}
        self.use_joint_constraints = getattr(Config, 'USE_JOINT_CONSTRAINTS', False)

    def _get_or_create_body(self, creature) -> RigidBody:
        """
        Return the RigidBody for this creature, creating it on first access.

        Each parallel environment (creature) owns its own RigidBody.
        Bodies are keyed by Python object id so there is zero shared state
        between environments, making ThreadPoolExecutor physics safe.

        Example:
            body = self._get_or_create_body(creature)  # own body, no races
            body.add_force(gravity)                     # affects only this env
        """
        cid = id(creature)
        if cid not in self._bodies:
            from config import Config
            m   = Config.BODY_MASS
            a, b, c_dim = Config.BODY_DIMENSIONS
            I_diag = np.array([
                (1/12) * m * (b * b + c_dim * c_dim),
                (1/12) * m * (a * a + c_dim * c_dim),
                (1/12) * m * (a * a + b * b),
            ])
            self._bodies[cid] = RigidBody(
                pos=np.array([float(creature.pos[0]),
                               float(creature.pos[1]),
                               float(creature.pos[2])]),
                orientation=Quaternion.identity(),
                linear_vel=np.zeros(3),
                angular_vel=np.zeros(3),
                mass=m,
                inertia_tensor=np.diag(I_diag),
            )
        return self._bodies[cid]

    def apply_motor_torques(self, creature, motor_torques):
        """
        Apply motor torques and integrate rigid body dynamics (quaternion-based orientation, Euler equations).
        See docs/PHYSICS.md for implementation details.
        
        PHASE 1 FEATURE: Actuator response lag (first-order filter)
        - If ACTUATOR_RESPONSE_TIME > 0: simulates servo lag
        - τ_applied = τ_applied + (τ_commanded - τ_applied) * (dt / response_time)
        - Otherwise: direct torque application (legacy)
        
        Returns:
            com_pos: (3,) new center of mass position
            stability_metrics: dict with balance info
        """
        
        motor_torques = torch.clamp(motor_torques, -self.max_torque, self.max_torque)
        
        # ===== PHASE 1: Actuator Response Lag =====
        if self.actuator_response_time > 0.0:
            # Initialize per-creature actuator state if needed
            if not hasattr(creature, '_actuator_state'):
                creature._actuator_state = torch.zeros_like(motor_torques)
            
            # First-order response: τ_applied += (τ_commanded - τ_applied) * (dt / τ_response)
            response_factor = (self.dt / self.actuator_response_time)
            creature._actuator_state.copy_(
                creature._actuator_state + (motor_torques - creature._actuator_state) * response_factor
            )
            # Use lagged torques for physics
            motor_torques = creature._actuator_state.clone()
        
        # Joint dynamics run on CPU via PyTorch matmuls.
        # WHY NOT CUDA here: launching any CUDA kernel costs ~10-30 µs of
        # overhead.  For 12 joint values that is more than the actual math.
        # Worse, the old kernel ran joint updates TWICE (GPU then CPU again),
        # wasting work.  Batched GPU kernels are still used in step_batch()
        # for the 1000+ environment case where launch overhead amortises.
        return self._update_joint_dynamics_cpu(creature, motor_torques)
    
    def _update_joint_dynamics_cpu(self, creature, motor_torques):
        """
        Joint dynamics + contact physics for one creature (CPU PyTorch).

        Per-creature RigidBody: each environment gets its own body via
        _get_or_create_body().  The old shared self.body caused race
        conditions in ThreadPoolExecutor and wrong physics (one env's
        step overwrote another env's rigid-body state).
        """
        # Fetch (or lazily create) this creature's dedicated rigid body.
        body = self._get_or_create_body(creature)

        # ---- 1. Joint velocity + angle integration ----
        creature.joint_velocities.copy_(
            creature.joint_velocities + (motor_torques.squeeze() - self.joint_damping * creature.joint_velocities) * self.dt
        )
        creature.joint_velocities.copy_(torch.clamp(creature.joint_velocities, -self.max_joint_velocity, self.max_joint_velocity))
        creature.joint_angles.copy_(creature.joint_angles + creature.joint_velocities * self.dt)
        creature.joint_angles.copy_(torch.clamp(creature.joint_angles, -math.pi, math.pi))

        # ---- 2. Forward kinematics → foot positions ----
        from .entity import compute_foot_positions
        foot_positions = compute_foot_positions(creature.joint_angles, creature.orientation, self.segment_length)

        # ---- 3. Contact detection + spring-damper forces (vectorised) ----
        # Replace the 4-iteration Python loop with NumPy array ops – same
        # arithmetic, ~4× less Python overhead per creature.
        energy_consumed = 0.0
        contact_force_total = np.zeros(3)

        # foot_positions is a (4, 3) torch tensor; one .numpy() call is cheaper
        # than four float() casts inside a Python loop.
        foot_z_np   = foot_positions[:, 2].detach().cpu().numpy().astype(np.float64)  # (4,)
        threshold   = self.ground_level + self.foot_height_threshold
        contact_mask_np = foot_z_np <= threshold              # (4,) bool

        creature.foot_contact.copy_(
            torch.from_numpy(contact_mask_np.astype(np.float32)).to(self.device)
        )
        num_contacts = int(contact_mask_np.sum())

        if num_contacts > 0:
            bvz  = float(body.linear_vel[2])
            pens = np.maximum(0.0, self.ground_level - foot_z_np) * contact_mask_np  # (4,)
            spring  = self.contact_stiffness * pens
            damper  = self.contact_damping   * bvz
            resti   = (self.contact_restitution * (-bvz)
                       * spring / np.maximum(pens, 0.001))
            # Mask non-contacting feet: when bvz < 0 (body falling), damper < 0 and
            # `spring - damper` would be positive even where spring == 0 (no contact).
            contact_force_total[2] = float(
                np.sum(np.maximum(0.0, spring - damper + resti) * contact_mask_np)
            )

        # ---- 4. Friction model ----
        if num_contacts > 0 and contact_force_total[2] > 0:
            normal_force = contact_force_total[2]
            foot_vel_horizontal = np.sqrt(body.linear_vel[0]**2 + body.linear_vel[1]**2)

            if self.friction_model == "coulomb+viscous":
                friction_force = self._compute_friction_force_coulomb_viscous(normal_force, foot_vel_horizontal)
            elif self.friction_model == "coulomb":
                friction_force = self.friction_coeff_kinetic * normal_force
            else:
                friction_force = self.friction_coeff * normal_force if hasattr(self, 'friction_coeff') else 0.0

            if self.use_friction_cones and foot_vel_horizontal > self.friction_slip_threshold:
                friction_force = min(friction_force, self.friction_coeff_kinetic * normal_force)
                direction = np.array([body.linear_vel[0], body.linear_vel[1], 0.0]) / (foot_vel_horizontal + 1e-8)
                friction_vector = -friction_force * direction / num_contacts
                damping_vector = -self.friction_cone_damping * np.array([body.linear_vel[0], body.linear_vel[1], 0.0]) / num_contacts
                contact_force_total[:2] += friction_vector[:2] + damping_vector[:2]
            elif foot_vel_horizontal > self.friction_slip_threshold:
                direction = np.array([body.linear_vel[0], body.linear_vel[1], 0.0]) / (foot_vel_horizontal + 1e-8)
                friction_vector = -friction_force * direction / num_contacts
                contact_force_total[:2] += friction_vector[:2]

        if num_contacts > 0:
            contact_force_total[2] /= num_contacts

        # ---- 5. Energy tracking (Phase 4) ----
        if self.track_energy:
            torques_sq = motor_torques.squeeze() if motor_torques.dim() > 1 else motor_torques
            vels_sq    = creature.joint_velocities.squeeze() if creature.joint_velocities.dim() > 1 else creature.joint_velocities
            mechanical_power  = torch.sum(torch.abs(torques_sq * vels_sq)).item()
            electrical_power  = mechanical_power / max(self.motor_efficiency, 0.01)
            energy_consumed   = electrical_power * self.dt
            if not hasattr(creature, '_total_energy_consumed'):
                creature._total_energy_consumed = 0.0
            creature._total_energy_consumed += energy_consumed

        # ---- 6. Apply forces + integrate rigid body ----
        gravity_factor = 1.0 - min(num_contacts, 4) / 4.0
        body.add_force(np.array([0, 0, -self.gravity * body.mass * gravity_factor]))
        body.add_force(contact_force_total)
        body.integrate(self.dt, gravity=0)

        # ---- 7. Joint constraints (Phase 5, optional) ----
        if self.use_joint_constraints and id(creature) in self.joint_constraints:
            for joint in self.joint_constraints[id(creature)]:
                joint.apply_constraint(body, None, self.dt)

        # ---- 8. Sync creature tensors from rigid body (in-place) ----
        creature.pos.copy_(torch.as_tensor(body.pos, device=self.device, dtype=self.dtype))
        pitch, yaw, roll = body.orientation.to_euler()
        creature.orientation[0] = float(pitch)
        creature.orientation[1] = float(yaw)
        creature.orientation[2] = float(roll)
        creature.velocity.copy_(torch.as_tensor(body.linear_vel, device=self.device, dtype=self.dtype))

        stability_metrics = {
            'foot_positions' : foot_positions,
            'com_pos'        : creature.pos,
            'num_contacts'   : num_contacts,
            'pitch'          : creature.orientation[0],
            'roll'           : creature.orientation[2],
            'angular_vel'    : torch.tensor(body.angular_vel, device=self.device, dtype=self.dtype),
            'energy_consumed': energy_consumed,
        }
        return creature.pos, stability_metrics

    def _compute_friction_force_coulomb_viscous(self, normal_force: float, slip_velocity: float) -> float:
        """
        PHASE 2: Compute friction force using Coulomb + viscous damping model.
        
        Model:
            F_friction = μ_kinetic * N + η * v_slip  (kinetic)
            
        Where:
            - μ_kinetic: kinetic friction coefficient (lower than static)
            - N: normal contact force
            - η: viscous damping coefficient
            - v_slip: horizontal slip velocity
        
        Returns:
            friction_force: scalar magnitude of friction (always >= 0)
        """
        if slip_velocity < self.friction_slip_threshold:
            # Low slip: use static friction coefficient (higher)
            coulomb_part = self.friction_coeff_static * normal_force
        else:
            # Active slip: use kinetic friction coefficient (lower)
            coulomb_part = self.friction_coeff_kinetic * normal_force
        
        # Add viscous damping term
        viscous_part = self.friction_viscous_damping * slip_velocity
        
        return coulomb_part + viscous_part
    
    def configure_joint_constraints(self, creature, joint_configs):
        """
        PHASE 5: Configure realistic joint constraints for a creature.
        
        Enables joint limits, damping, and friction on top of motor torque control.
        
        Args:
            creature: Creature object to apply joint constraints to
            joint_configs: List of JointConfig objects (from joint_constraints.py)
            
        Example:
            from source.joint_constraints import QuadrupedJointSetup
            
            joints = QuadrupedJointSetup.create_quadruped_joints(device, dtype)
            physics_engine.configure_joint_constraints(creature, joints)
        
        Returns:
            True if constraints successfully configured, False otherwise
        """
        try:
            from source.joint_constraints import JointConstraint
            
            creature_id = id(creature)
            joint_constraints = []
            
            for joint_config in joint_configs:
                constraint = JointConstraint(joint_config, device=self.device, dtype=self.dtype)
                joint_constraints.append(constraint)
            
            # Store joint constraints for this creature
            self.joint_constraints[creature_id] = joint_constraints
            self.use_joint_constraints = True  # Enable constraint processing
            
            return True
        except Exception as e:
            print(f"[Physics] Failed to configure joint constraints: {e}")
            return False
    
    def compute_balance_reward(self, com_pos, stability_metrics, motor_torques, goal_pos):
        """
        Compute reward: balance (primary) > goal-reaching (secondary) > efficiency (tertiary).
        See docs/EVALUATION.md for reward structure details.
        
        PHASE 4: Energy-aware rewards
        - If TRACK_ENERGY_CONSUMPTION enabled: include energy penalty in reward
        - Encourages efficient motor control without explicit power sensing
        """
        # Keep as tensors for autograd compatibility
        pitch = stability_metrics['pitch']
        roll = stability_metrics['roll']
        
        # Balance reward: reward staying upright (small pitch/roll)
        # Quadratic penalty grows with tilt angle: discourages falling
        balance_reward = torch.exp(-0.5 * (pitch**2 + roll**2) / (0.1**2))  # Strong reward for upright
        
        # Contact bonus: prefer multiple foot contacts
        num_contacts = stability_metrics['num_contacts']
        contact_reward = min(num_contacts, 4) * self.contact_reward
        
        # Hard penalty for excessive tilt
        stability_penalty = 0.0
        if abs(pitch) > self.max_pitch_roll or abs(roll) > self.max_pitch_roll:
            stability_penalty = -self.tilt_penalty
        
        # Goal reaching reward (secondary, modulated by stability)
        # Use x-y horizontal plane (z is the vertical/UP axis in this physics engine)
        goal_relative = goal_pos - com_pos
        goal_xy = torch.stack([goal_relative[0], goal_relative[1]])
        com_dist = torch.norm(goal_xy)
        
        # Goal reward: reach goal while maintaining balance
        stability_factor = max(0.0, 1.0 - (abs(pitch) + abs(roll)) / (2 * self.max_pitch_roll))
        goal_reward = torch.exp(-com_dist) * stability_factor
        
        # Energy penalty (discourage excessive torques to encourage efficiency)
        torque_magnitude = torch.norm(motor_torques)
        energy_cost = torque_magnitude * self.energy_penalty
        
        # Phase 4: Add tracked energy penalty if enabled
        tracked_energy_penalty = 0.0
        if self.track_energy and 'energy_consumed' in stability_metrics:
            # Convert tracked energy (joules) to penalty
            # Typical range: 0-10 joules per step → factor to make comparable to other rewards
            tracked_energy_penalty = stability_metrics['energy_consumed'] * 0.001
        
        # Total reward: balance is primary, goal-reaching is secondary
        # Order: stability (balance + contacts + penalties) > goal reaching > efficiency
        total_reward = balance_reward + contact_reward + stability_penalty + goal_reward - energy_cost - tracked_energy_penalty
        
        return total_reward, com_dist
    
    def _compute_reward(self, creature, motor_torques, goal_pos):
        """High-level reward computation for quadruped balance task."""
        com_pos, stability_metrics = self.apply_motor_torques(creature, motor_torques)
        reward, com_dist = self.compute_balance_reward(com_pos, stability_metrics, motor_torques, goal_pos)
        return reward, com_dist, stability_metrics
    
    def step_batch(self, creatures_batch, motor_torques_batch, goal_pos_batch):
        """
        Vectorized physics step for multiple creatures (GPU-accelerated).

        PHASE 4B: Vectorized physics engine - processes entire batch on GPU.
        Enables 100-1000x speedup for 1000+ parallel environments.

        Args:
            creatures_batch: list of creatures (tensors already on GPU)
            motor_torques_batch: (num_envs, 12) tensor of motor commands
            goal_pos_batch: (num_envs, 3) tensor of per-creature goal positions

        Returns:
            rewards_batch: (num_envs,) tensor of rewards
            distances_batch: (num_envs,) tensor of distances to goal
            metrics_batch: dict of batched stability metrics
        """
        from config import Config

        num_envs = len(creatures_batch)

        # If GPU available and vectorization enabled, use batched kernels
        if HAS_CUDA and self.device.type == 'cuda' and getattr(Config, 'VECTORIZED_PHYSICS', False):
            return self._step_batch_gpu(creatures_batch, motor_torques_batch, goal_pos_batch)
        else:
            # Fallback: process sequentially (slower but works on CPU)
            rewards = []
            distances = []
            metrics_list = []

            for i, creature in enumerate(creatures_batch):
                reward, distance, metrics = self._compute_reward(
                    creature, motor_torques_batch[i], goal_pos_batch[i]
                )
                rewards.append(reward)
                distances.append(distance)
                metrics_list.append(metrics)
            
            rewards_batch = torch.stack(rewards)
            distances_batch = torch.stack(distances)
            
            # Aggregate metrics
            metrics_batch = {
                'num_contacts': torch.tensor([m['num_contacts'] for m in metrics_list], device=self.device),
                'pitch': torch.stack([m.get('pitch', torch.tensor(0.0, device=self.device)) for m in metrics_list]),
                'roll': torch.stack([m.get('roll', torch.tensor(0.0, device=self.device)) for m in metrics_list]),
            }
            
            return rewards_batch, distances_batch, metrics_batch
    
    def _batch_forward_kinematics(
        self,
        all_joint_angles: torch.Tensor,   # (N, 12)
        all_orientations: torch.Tensor,   # (N,  3)  [pitch, yaw, roll]
        segment_length: float,
    ) -> torch.Tensor:                    # (N,  4, 3)
        """
        Batched forward kinematics: N creatures × 4 legs → (N, 4, 3) foot positions.

        Everything runs as a single GPU pass (no Python loop over environments):
          1. Build (N, 3, 3) rotation matrices from Euler angles using batched ops.
          2. Compute cumulative pitch-plane foot offsets  →  (N, 3)  per leg.
          3. Rotate local offset + hip offset into world frame via torch.bmm.

        Example – N=16 envs:
          Before (sequential):  16 × 4 = 64 separate  forward_kinematics_leg() calls.
          After  (batched):      4 torch.bmm calls over (16, 3, 3) tensors (one per leg).
        """
        N      = all_joint_angles.shape[0]
        device = all_joint_angles.device
        dtype  = all_joint_angles.dtype
        seg    = segment_length

        pitch = all_orientations[:, 0]   # (N,)
        yaw   = all_orientations[:, 1]
        roll  = all_orientations[:, 2]

        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw),   torch.sin(yaw)
        cr, sr = torch.cos(roll),  torch.sin(roll)
        zero   = torch.zeros(N, device=device, dtype=dtype)
        one    = torch.ones (N, device=device, dtype=dtype)

        # Rx – pitch rotation around X axis
        Rx = torch.stack([
            torch.stack([one,  zero, zero], dim=1),
            torch.stack([zero,  cp,   -sp], dim=1),
            torch.stack([zero,  sp,    cp], dim=1),
        ], dim=1)  # (N, 3, 3)

        # Ry – roll rotation around Y axis
        Ry = torch.stack([
            torch.stack([cr,   zero,  sr], dim=1),
            torch.stack([zero,  one, zero], dim=1),
            torch.stack([-sr,  zero,  cr], dim=1),
        ], dim=1)

        # Rz – yaw rotation around Z axis
        Rz = torch.stack([
            torch.stack([cy,  -sy, zero], dim=1),
            torch.stack([sy,   cy, zero], dim=1),
            torch.stack([zero, zero, one], dim=1),
        ], dim=1)

        R = torch.bmm(torch.bmm(Rz, Ry), Rx)  # (N, 3, 3)

        from .entity import _HIP_OFFSETS_RAW
        feet = []
        for leg_idx in range(4):
            la = all_joint_angles[:, leg_idx * 3 : leg_idx * 3 + 3]  # (N, 3)
            a1 = la[:, 0]
            a2 = la[:, 0] + la[:, 1]
            a3 = la[:, 0] + la[:, 1] + la[:, 2]

            c1, s1 = torch.cos(a1), torch.sin(a1)
            c2, s2 = torch.cos(a2), torch.sin(a2)
            c3, s3 = torch.cos(a3), torch.sin(a3)

            # Pitch-plane cumulative offsets (x=0 for each segment)
            p1 = torch.stack([zero, -seg * c1, -seg * s1], dim=1)
            p2 = p1 + torch.stack([zero, -seg * c2, -seg * s2], dim=1)
            p3 = p2 + torch.stack([zero, -seg * c3, -seg * s3], dim=1)  # (N, 3)

            # Rotate foot offset and hip offset into world frame  →  (N, 3)
            p3_world  = torch.bmm(R, p3.unsqueeze(-1)).squeeze(-1)
            hip       = torch.tensor(
                _HIP_OFFSETS_RAW[leg_idx], dtype=dtype, device=device
            ).unsqueeze(0).expand(N, -1)  # (N, 3)
            hip_world = torch.bmm(R, hip.contiguous().unsqueeze(-1)).squeeze(-1)

            feet.append(hip_world + p3_world)

        return torch.stack(feet, dim=1)  # (N, 4, 3)

    def _step_batch_gpu(self, creatures_batch, motor_torques_batch, goal_pos_batch):
        """
        GPU-batched physics using PyTorch vectorised ops (no Numba kernels).

        All N environments are processed together in a few tensor calls instead
        of a sequential Python for-loop.  PyTorch dispatches to cuBLAS / cuDNN
        and selects the optimal grid/block layout automatically — eliminating the
        "Grid size 1" under-utilisation warning that appears when launching a
        Numba kernel with only 12 work items (1 quadruped's joints).

        Per-step complexity:  O(1)  CUDA launches regardless of N:
          • Joint dynamics  – 3 elementwise ops  on (N, 12) tensors
          • Forward kinematics – 4 × bmm  on (N, 3, 3)
          • Contact detection – 1 comparison + sum  on (N, 4)
          • Spring-damper     – 5 elementwise ops  on (N, 4)
          • RigidBody + reward – per-creature (NumPy; unavoidable until full tensor refactor)
        """
        num_envs = len(creatures_batch)
        
        # Clamp motor torques
        motor_torques_batch = torch.clamp(motor_torques_batch, -self.max_torque, self.max_torque)
        
        # Phase 1: Actuator response lag (vectorized)
        if self.actuator_response_time > 0.0:
            for i, creature in enumerate(creatures_batch):
                if not hasattr(creature, '_actuator_state'):
                    creature._actuator_state = torch.zeros_like(motor_torques_batch[i])
                response_factor = (self.dt / self.actuator_response_time)
                creature._actuator_state.copy_(
                    creature._actuator_state + (motor_torques_batch[i] - creature._actuator_state) * response_factor
                )
                motor_torques_batch[i] = creature._actuator_state.clone()
        
        # ── 1. Joint dynamics – fully batched, stays on GPU ──────────────────
        # Stack across all envs: (N, 12) tensors on the compute device.
        # Three elementwise ops replace N sequential RigidBody updates.
        all_vels   = torch.stack([c.joint_velocities for c in creatures_batch])  # (N, 12)
        all_angles = torch.stack([c.joint_angles     for c in creatures_batch])  # (N, 12)

        all_vels   = all_vels + (motor_torques_batch - self.joint_damping * all_vels) * self.dt
        all_vels   = torch.clamp(all_vels, -self.max_joint_velocity, self.max_joint_velocity)
        all_angles = all_angles + all_vels * self.dt
        # Wrap angles to [-π, π] without a Python loop
        all_angles = torch.remainder(all_angles + math.pi, 2.0 * math.pi) - math.pi

        for i, creature in enumerate(creatures_batch):
            creature.joint_velocities.copy_(all_vels[i])
            creature.joint_angles.copy_(all_angles[i])

        # ── 2. Batched forward kinematics → (N, 4, 3) foot positions ─────────
        # One GPU pass for all N environments × 4 legs (4 × bmm on (N,3,3)).
        all_orientations     = torch.stack([c.orientation for c in creatures_batch])  # (N, 3)
        foot_positions_batch = self._batch_forward_kinematics(
            all_angles, all_orientations, self.segment_length
        )  # (N, 4, 3)

        # ── 3. Contact detection + spring-damper forces – batched ─────────────
        foot_z       = foot_positions_batch[:, :, 2]                              # (N, 4)
        contact_mask = (foot_z <= self.ground_level + self.foot_height_threshold).float()  # (N, 4)
        num_contacts = contact_mask.sum(dim=1)                                    # (N,)

        for i, creature in enumerate(creatures_batch):
            creature.foot_contact.copy_(contact_mask[i])

        # Body z-velocity for each creature (needed for spring-damper damper term)
        body_vz = torch.tensor(
            [self._get_or_create_body(c).linear_vel[2] for c in creatures_batch],
            device=self.device, dtype=self.dtype,
        )  # (N,)

        penetrations = torch.clamp(self.ground_level - foot_z, min=0.0) * contact_mask  # (N, 4)
        spring       = self.contact_stiffness  * penetrations
        damper       = self.contact_damping    * body_vz.unsqueeze(1)
        restitution  = (
            self.contact_restitution * (-body_vz.unsqueeze(1))
            * spring / torch.clamp(penetrations, min=1e-4)
            * contact_mask
        )
        # Mask non-contacting feet: when body_vz < 0 (body falling), damper < 0 and
        # `spring - damper` would be positive for feet where spring == 0 (no contact).
        contact_forces_per_foot = torch.clamp(spring - damper + restitution, min=0.0) * contact_mask
        # Average over contacting feet per environment
        contact_force_z = (
            contact_forces_per_foot.sum(dim=1)
            / torch.clamp(num_contacts, min=1.0)
        )  # (N,)

        # ── 4. Per-creature rigid-body integration + reward ───────────────────
        # RigidBody is NumPy-based; full tensor refactor is a separate task.
        # This loop is now small: joint dynamics and FK have already been done above.
        rewards_batch   = []
        distances_batch = []

        for i, creature in enumerate(creatures_batch):
            body       = self._get_or_create_body(creature)
            n_contacts = int(num_contacts[i].item())
            fz         = float(contact_force_z[i].item())

            gravity_factor = 1.0 - min(n_contacts, 4) / 4.0
            body.add_force(np.array([0.0, 0.0, -self.gravity * body.mass * gravity_factor]))
            body.add_force(np.array([0.0, 0.0, fz]))
            body.integrate(self.dt, gravity=0)

            creature.pos.copy_(
                torch.as_tensor(body.pos, device=self.device, dtype=self.dtype)
            )
            pitch, yaw, roll = body.orientation.to_euler()
            creature.orientation[0] = float(pitch)
            creature.orientation[1] = float(yaw)
            creature.orientation[2] = float(roll)

            reward, distance = self.compute_balance_reward(
                creature.pos,
                {
                    'pitch'          : creature.orientation[0],
                    'roll'           : creature.orientation[2],
                    'num_contacts'   : n_contacts,
                    'energy_consumed': 0.0,
                },
                motor_torques_batch[i],
                goal_pos_batch[i],  # per-creature goal
            )
            rewards_batch.append(reward)
            distances_batch.append(distance)

        rewards_batch   = torch.stack(rewards_batch)
        distances_batch = torch.stack(distances_batch)

        metrics_batch = {
            'num_contacts': num_contacts,
            'pitch': torch.stack([c.orientation[0] for c in creatures_batch]),
            'roll' : torch.stack([c.orientation[2] for c in creatures_batch]),
        }

        return rewards_batch, distances_batch, metrics_batch