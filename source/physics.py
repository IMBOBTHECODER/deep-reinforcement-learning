"""
Advanced Physics Engine for Quadruped Robot.
See docs/PHYSICS.md for detailed feature documentation.

Key Classes: Quaternion (gimbal-lock-free), RigidBody (full dynamics), ContactManifold (impulse resolution).
"""

import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


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
        """Convert quaternion back to Euler angles (pitch, yaw, roll)."""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (self.w * self.y - self.z * self.x)
        sinp = max(-1, min(1, sinp))
        pitch = math.asin(sinp)
        
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return pitch, yaw, roll
    
    def normalize(self):
        """Normalize quaternion to unit length."""
        mag = math.sqrt(self.w * self.w + self.x * self.x + 
                        self.y * self.y + self.z * self.z)
        if mag > 1e-6:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by this quaternion."""
        u = np.array([self.x, self.y, self.z])
        term1 = 2 * self.w * np.cross(u, v)
        term2 = 2 * np.cross(u, np.cross(u, v))
        return v + term1 + term2
    
    def __mul__(self, other):
        """Quaternion multiplication (non-commutative)."""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        raise TypeError("Can only multiply Quaternion with Quaternion")
    
    def conjugate(self):
        """Return conjugate (inverse for unit quaternions)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3×3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])


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
    
    def add_torque(self, torque: np.ndarray):
        """Add torque directly."""
        self.torque_accum += torque
    
    def clear_forces(self):
        """Clear accumulated forces and torques."""
        self.force_accum.fill(0)
        self.torque_accum.fill(0)
    
    def integrate(self, dt: float, gravity: float = 9.8):
        """
        Integrate rigid body dynamics using semi-implicit Euler.
        
        Includes full Euler equations with gyroscopic effects.
        """
        from config import Config
        
        # Linear motion
        accel = self.force_accum / self.mass
        accel[2] -= gravity
        
        self.linear_vel += accel * dt
        self.pos += self.linear_vel * dt
        
        # Angular motion
        R = self.orientation.to_rotation_matrix()
        I_world = R @ self.inertia_tensor @ R.T
        
        I_omega = I_world @ self.angular_vel
        gyroscopic_torque = np.cross(self.angular_vel, I_omega)
        
        try:
            I_inv = np.linalg.inv(I_world)
            angular_accel = I_inv @ (self.torque_accum - gyroscopic_torque)
        except np.linalg.LinAlgError:
            angular_accel = (self.torque_accum - gyroscopic_torque) / np.diag(self.inertia_tensor)
        
        self.angular_vel += angular_accel * dt
        
        # Clamp angular velocity
        ang_mag = np.linalg.norm(self.angular_vel)
        if ang_mag > Config.MAX_ANGULAR_VELOCITY:
            self.angular_vel = self.angular_vel * (Config.MAX_ANGULAR_VELOCITY / ang_mag)
        
        # Integrate quaternion
        omega_quat = Quaternion(0, self.angular_vel[0], self.angular_vel[1], self.angular_vel[2])
        q_deriv = self.orientation * omega_quat
        q_deriv.w *= 0.5 * dt
        q_deriv.x *= 0.5 * dt
        q_deriv.y *= 0.5 * dt
        q_deriv.z *= 0.5 * dt
        
        self.orientation = Quaternion(
            self.orientation.w + q_deriv.w,
            self.orientation.x + q_deriv.x,
            self.orientation.y + q_deriv.y,
            self.orientation.z + q_deriv.z
        )
        self.orientation.normalize()
        
        self.clear_forces()


@dataclass
class ContactManifold:
    """Contact point between two bodies with impulse resolution."""
    body_a: RigidBody
    body_b: Optional[RigidBody]  # None if ground
    
    pos: np.ndarray              # Contact position (3D)
    normal: np.ndarray           # Contact normal (3D)
    penetration: float           # Penetration depth
    
    accumulated_normal_impulse: float = 0.0
    accumulated_tangent_impulse: np.ndarray = None
    
    restitution: float = None  # Bounciness
    friction_coeff: float = None  # Coulomb friction
    
    def __post_init__(self):
        from config import Config
        
        if self.accumulated_tangent_impulse is None:
            self.accumulated_tangent_impulse = np.zeros(2)
        
        if self.restitution is None:
            self.restitution = Config.CONTACT_RESTITUTION
        
        if self.friction_coeff is None:
            self.friction_coeff = Config.GROUND_FRICTION_COEFFICIENT
    
    def solve_impulse(self, dt: float, bias: float = 0.2):
        """
        Resolve contact using sequential impulses with Baumgarte stabilization and friction (Box2D style).
        
        Implements:
        1. Effective mass scaling: impulse = velocity_constraint / inv_mass_sum
        2. Baumgarte bias: adds positional correction based on penetration depth
        3. Friction cone: accumulated impulses clamped to mu * normal_impulse
        4. Sequential impulse accumulation with separating guard
        """
        # Relative velocity at contact
        v_a = self.body_a.linear_vel + np.cross(self.body_a.angular_vel, self.pos - self.body_a.pos)
        
        if self.body_b is not None:
            v_b = self.body_b.linear_vel + np.cross(self.body_b.angular_vel, self.pos - self.body_b.pos)
        else:
            v_b = np.zeros(3)
        
        rel_vel = v_b - v_a
        vel_along_normal = np.dot(rel_vel, self.normal)
        
        # Build tangent space
        tangent1 = np.cross([0, 0, 1], self.normal)
        if np.linalg.norm(tangent1) < 0.1:
            tangent1 = np.cross([1, 0, 0], self.normal)
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(self.normal, tangent1)
        
        # Compute effective mass (inverse mass sum for linear contact)
        inv_mass_a = 1.0 / self.body_a.mass if self.body_a.mass > 0 else 0
        inv_mass_b = (1.0 / self.body_b.mass if self.body_b.mass > 0 else 0) if self.body_b is not None else 0
        inv_mass_sum = inv_mass_a + inv_mass_b
        
        # Guard: skip if no effective mass (both bodies infinite mass)
        if inv_mass_sum < 1e-6:
            return
        
        # Baumgarte stabilization bias: correct positional error via velocity constraint
        baumgarte_bias = bias * max(0, self.penetration - 0.01) / dt if dt > 0 else 0
        
        # Compute velocity constraint for normal direction
        # Include both restitution (bounciness) and Baumgarte bias (penetration correction)
        velocity_constraint = -(1 + self.restitution) * vel_along_normal - baumgarte_bias
        
        # Sequential impulse with proper mass scaling
        # impulse_magnitude = velocity_constraint / inv_mass_sum
        old_normal_impulse = self.accumulated_normal_impulse
        self.accumulated_normal_impulse = max(0, self.accumulated_normal_impulse + velocity_constraint / inv_mass_sum)
        delta_normal_impulse = self.accumulated_normal_impulse - old_normal_impulse
        
        # Apply normal impulse (scaled by mass inverse)
        impulse_normal = delta_normal_impulse * self.normal
        self.body_a.linear_vel -= impulse_normal * inv_mass_a
        if self.body_b is not None:
            self.body_b.linear_vel += impulse_normal * inv_mass_b
        
        # Friction (tangent plane) - only apply if in contact (normal impulse > 0)
        if self.accumulated_normal_impulse > 0:
            vel_t1 = np.dot(rel_vel, tangent1)
            vel_t2 = np.dot(rel_vel, tangent2)
            
            # Tangent impulses (dry friction model)
            friction_limit = self.friction_coeff * self.accumulated_normal_impulse
            
            # Tangent 1
            velocity_constraint_t1 = -vel_t1 / inv_mass_sum if inv_mass_sum > 0 else 0
            old_t1 = self.accumulated_tangent_impulse[0]
            self.accumulated_tangent_impulse[0] = np.clip(old_t1 + velocity_constraint_t1, -friction_limit, friction_limit)
            delta_t1 = self.accumulated_tangent_impulse[0] - old_t1
            
            # Tangent 2
            velocity_constraint_t2 = -vel_t2 / inv_mass_sum if inv_mass_sum > 0 else 0
            old_t2 = self.accumulated_tangent_impulse[1]
            self.accumulated_tangent_impulse[1] = np.clip(old_t2 + velocity_constraint_t2, -friction_limit, friction_limit)
            delta_t2 = self.accumulated_tangent_impulse[1] - old_t2
            
            # Apply friction impulses (mass-scaled)
            impulse_friction = delta_t1 * tangent1 + delta_t2 * tangent2
            self.body_a.linear_vel -= impulse_friction * inv_mass_a
            if self.body_b is not None:
                self.body_b.linear_vel += impulse_friction * inv_mass_b


class PhysicsEngine:
    """
    Advanced physics engine with rigid body dynamics, quaternion orientation, and realistic contacts.
    Features: gravity integration, spring-damper contacts, joint velocity clamping, agent-centered world.
    IMPROVEMENTS: Quaternion-based orientation (no gimbal lock), inertia tensor, Euler equations.
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
        
        # Contact physics (spring-damper model with Coulomb friction)
        self.ground_level = Config.GROUND_LEVEL
        self.foot_height_threshold = Config.FOOT_HEIGHT_THRESHOLD
        self.contact_stiffness = Config.CONTACT_STIFFNESS
        self.contact_damping = Config.CONTACT_DAMPING
        self.friction_coeff = Config.GROUND_FRICTION_COEFFICIENT
        
        # Reward parameters
        self.com_distance_threshold = Config.COM_DISTANCE_THRESHOLD
        self.contact_reward = Config.CONTACT_REWARD
        self.energy_penalty = Config.ENERGY_PENALTY
        self.tilt_penalty = Config.TILT_PENALTY
        
        self.max_pitch_roll = Config.MAX_PITCH_ROLL
        
        # RIGID BODY: Torso with full dynamics (mass, inertia, orientation)
        m = Config.BODY_MASS
        a, b, c = Config.BODY_DIMENSIONS
        I_diag = np.array([
            (1/12) * m * (b*b + c*c),  # I_xx
            (1/12) * m * (a*a + c*c),  # I_yy
            (1/12) * m * (a*a + b*b)   # I_zz
        ])
        I_tensor = np.diag(I_diag)
        
        self.body = RigidBody(
            pos=np.array([0.0, 0.0, Config.BODY_INITIAL_HEIGHT]),
            orientation=Quaternion.identity(),
            linear_vel=np.zeros(3),
            angular_vel=np.zeros(3),
            mass=m,
            inertia_tensor=I_tensor
        )
        
        self.agent_local_pos = torch.zeros(3, device=device, dtype=dtype)
        self.atanh_eps = Config.ATANH_EPSILON
        self.log_eps = Config.LOG_EPSILON
    
    def apply_motor_torques(self, creature, motor_torques):
        """
        Apply motor torques and integrate rigid body dynamics (quaternion-based orientation, Euler equations).
        See docs/PHYSICS.md for implementation details.
        
        Returns:
            com_pos: (3,) new center of mass position
            stability_metrics: dict with balance info
        """
        motor_torques = torch.clamp(motor_torques, -self.max_torque, self.max_torque)
        
        # Update joint dynamics
        creature.joint_velocities.copy_(
            creature.joint_velocities + (motor_torques - self.joint_damping * creature.joint_velocities) * self.dt
        )
        
        creature.joint_velocities.copy_(torch.clamp(creature.joint_velocities, -self.max_joint_velocity, self.max_joint_velocity))
        
        creature.joint_angles.copy_(creature.joint_angles + creature.joint_velocities * self.dt)
        creature.joint_angles.copy_(torch.clamp(creature.joint_angles, -math.pi, math.pi))
        
        # Compute foot positions via forward kinematics
        from .entity import compute_foot_positions
        
        foot_positions = compute_foot_positions(creature.joint_angles, creature.orientation, self.segment_length)
        
        # Note: Agent COM is creature.pos (rigid body center of mass)
        # Kinematic COM from joint forward kinematics is not used for reward/control
        # (creature.pos is the actual integrated body position from physics)
        
        # ADVANCED: Detect contacts and apply forces to rigid body
        num_contacts = 0
        contact_force_total = np.zeros(3)
        
        for foot_idx in range(4):
            foot_z = float(foot_positions[foot_idx, 2])
            
            if foot_z <= self.ground_level + self.foot_height_threshold:
                num_contacts += 1
                creature.foot_contact[foot_idx] = 1.0
                
                # Spring-damper contact model (with Coulomb friction)
                penetration = max(0, self.ground_level - foot_z)
                contact_normal_force = self.contact_stiffness * penetration
                contact_damper_force = self.contact_damping * float(self.body.linear_vel[2])
                
                # Unilateral contact: force cannot pull (no adhesion)
                contact_force_z = max(0.0, contact_normal_force - contact_damper_force)
                contact_force_total[2] += contact_force_z
            else:
                creature.foot_contact[foot_idx] = 0.0
        
        # Normalize contact force by contacting feet
        if num_contacts > 0:
            contact_force_total[2] /= num_contacts
        
        # Apply forces to rigid body (contact-dependent gravity)
        gravity_factor = 1.0 - min(num_contacts, 4) / 4.0
        self.body.add_force(np.array([0, 0, -self.gravity * self.body.mass * gravity_factor]))
        self.body.add_force(contact_force_total)
        
        # Integrate rigid body dynamics
        self.body.integrate(self.dt, gravity=0)
        
        # Sync creature position with body COM (in-place update to preserve tensor references)
        creature.pos.copy_(torch.as_tensor(self.body.pos, device=self.device, dtype=self.dtype))
        
        # Update orientation from quaternion (element-wise, preserving tensor reference)
        pitch, yaw, roll = self.body.orientation.to_euler()
        creature.orientation[0] = float(pitch)
        creature.orientation[1] = float(yaw)
        creature.orientation[2] = float(roll)
        
        # Sync linear and angular velocity (in-place update to preserve tensor references)
        creature.velocity.copy_(torch.as_tensor(self.body.linear_vel, device=self.device, dtype=self.dtype))
        
        # Stability metrics
        stability_metrics = {
            'foot_positions': foot_positions,
            'com_pos': creature.pos,
            'num_contacts': num_contacts,
            'pitch': creature.orientation[0],
            'roll': creature.orientation[2],
            'angular_vel': torch.tensor(self.body.angular_vel, device=self.device, dtype=self.dtype),
        }
        
        return creature.pos, stability_metrics
    
    
    def compute_balance_reward(self, com_pos, stability_metrics, motor_torques, goal_pos):
        """
        Compute reward: balance (primary) > goal-reaching (secondary) > efficiency (tertiary).
        See docs/EVALUATION.md for reward structure details.
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
        goal_relative = goal_pos - com_pos
        goal_xy = torch.stack([goal_relative[0], goal_relative[2]])
        com_dist = torch.norm(goal_xy)
        
        # Goal reward: reach goal while maintaining balance
        stability_factor = max(0.0, 1.0 - (abs(pitch) + abs(roll)) / (2 * self.max_pitch_roll))
        goal_reward = torch.exp(-com_dist) * stability_factor
        
        # 2. Energy penalty (discourage excessive torques to encourage efficiency)
        torque_magnitude = torch.norm(motor_torques)
        energy_cost = torque_magnitude * self.energy_penalty
        
        # Total reward: balance is primary, goal-reaching is secondary
        # Order: stability (balance + contacts + penalties) > goal reaching > efficiency
        total_reward = balance_reward + contact_reward + stability_penalty + goal_reward - energy_cost
        
        return total_reward, com_dist
    
    def _compute_reward(self, creature, motor_torques, goal_pos):
        """High-level reward computation for quadruped balance task."""
        com_pos, stability_metrics = self.apply_motor_torques(creature, motor_torques)
        reward, com_dist = self.compute_balance_reward(com_pos, stability_metrics, motor_torques, goal_pos)
        return reward, com_dist, stability_metrics