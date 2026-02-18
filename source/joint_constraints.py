"""
Joint Constraint System for Physics Engine.
Supports realistic joint types: revolute, spherical, fixed, hinge2, prismatic.
See docs/JOINT_CONSTRAINTS.md for detailed documentation.
"""

import torch
import math
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List
from dataclasses import dataclass
from .physics import Quaternion


class JointType(Enum):
    """Joint constraint types."""
    REVOLUTE = "revolute"      # 1 DOF: rotation around single axis
    SPHERICAL = "spherical"    # 3 DOF: full 3D rotation
    FIXED = "fixed"            # 0 DOF: rigid connection
    HINGE2 = "hinge2"          # 2 DOF: rotation around two axes
    PRISMATIC = "prismatic"    # 1 DOF: linear motion along axis


@dataclass
class JointConfig:
    """Configuration for a joint constraint."""
    joint_type: JointType
    parent_link: str            # Name of parent rigid body
    child_link: str             # Name of child rigid body
    anchor_pos: np.ndarray      # (3,) position of joint in parent frame
    axis: np.ndarray            # (3,) rotation axis (normalized)
    axis2: Optional[np.ndarray] = None  # (3,) secondary axis for hinge2
    
    # Joint limits
    lower_limit: float = -math.pi   # rad (or meters for prismatic)
    upper_limit: float = math.pi    # rad (or meters for prismatic)
    
    # Joint properties
    stiffness: float = 0.0      # Spring stiffness (0=no restoring force)
    damping: float = 0.1        # Damping coefficient
    friction: float = 0.0       # Joint friction torque
    
    # Max force/torque
    max_force: float = 1000.0   # N (for prismatic)
    max_torque: float = 100.0   # N⋅m (for revolute/hinge2)


class JointConstraint:
    """
    Enforces joint constraints between two rigid bodies.
    Implements various joint types with limits, damping, and friction.
    """
    
    def __init__(self, config: JointConfig, device=None, dtype=None):
        self.config = config
        self.device = device
        self.dtype = dtype or torch.float32
        
        # Validate configuration
        self._validate_config()
        
        # Current joint state
        self.angle = torch.tensor(0.0, device=device, dtype=dtype)  # For 1-DOF joints
        self.angle2 = torch.tensor(0.0, device=device, dtype=dtype)  # For 2-DOF joints
        self.position = torch.tensor(0.0, device=device, dtype=dtype)  # For prismatic
        
        self.velocity = 0.0  # Angular or linear velocity
        self.velocity2 = 0.0  # Secondary angular velocity (hinge2)
        
        # Normalized axis vectors
        self.axis = torch.as_tensor(config.axis / np.linalg.norm(config.axis), 
                                    device=device, dtype=dtype)
        if config.axis2 is not None:
            self.axis2 = torch.as_tensor(config.axis2 / np.linalg.norm(config.axis2),
                                        device=device, dtype=dtype)
        else:
            self.axis2 = None
    
    def _validate_config(self):
        """Validate joint configuration."""
        if self.config.joint_type == JointType.HINGE2:
            if self.config.axis2 is None:
                raise ValueError("HINGE2 joint requires axis2")
        if self.config.joint_type == JointType.PRISMATIC:
            if self.config.lower_limit >= self.config.upper_limit:
                raise ValueError("lower_limit must be < upper_limit")
    
    def apply_constraint(self, body_a, body_b, dt):
        """
        Apply joint constraint forces/torques to both bodies.
        
        Args:
            body_a: Parent RigidBody
            body_b: Child RigidBody
            dt: Time step
        """
        if self.config.joint_type == JointType.REVOLUTE:
            self._apply_revolute_constraint(body_a, body_b, dt)
        elif self.config.joint_type == JointType.SPHERICAL:
            self._apply_spherical_constraint(body_a, body_b, dt)
        elif self.config.joint_type == JointType.FIXED:
            self._apply_fixed_constraint(body_a, body_b, dt)
        elif self.config.joint_type == JointType.HINGE2:
            self._apply_hinge2_constraint(body_a, body_b, dt)
        elif self.config.joint_type == JointType.PRISMATIC:
            self._apply_prismatic_constraint(body_a, body_b, dt)
    
    def _apply_revolute_constraint(self, body_a, body_b, dt):
        """
        REVOLUTE JOINT: 1 DOF rotation around single axis.
        
        Constraint: relative_rotation_around_axis = angle
        
        Applied forces:
        1. Restoring torque: tau = -stiffness * angle
        2. Damping: tau = -damping * velocity
        3. Friction: tau = -friction * sign(velocity)
        4. Limits: if angle exceeds limits, apply spring force back
        """
        # Compute relative rotation around axis
        relative_quaternion = body_b.orientation * Quaternion.inverse(body_a.orientation)
        
        # Extract rotation angle around axis
        angle_rad = self._extract_rotation_around_axis(relative_quaternion, self.axis)
        
        # Clamp to [-pi, pi]
        angle_rad = ((angle_rad + math.pi) % (2 * math.pi)) - math.pi
        
        # Update state
        self.velocity = (angle_rad - float(self.angle)) / dt
        self.angle.fill_(angle_rad)
        
        # Check limits
        if angle_rad < self.config.lower_limit:
            angle_rad = self.config.lower_limit
            self.velocity = 0.0
        elif angle_rad > self.config.upper_limit:
            angle_rad = self.config.upper_limit
            self.velocity = 0.0
        
        # Compute constraint torque
        tau = 0.0
        
        # 1. Restoring torque (spring)
        tau -= self.config.stiffness * angle_rad
        
        # 2. Limit penalty
        if angle_rad <= self.config.lower_limit:
            tau -= 10.0 * self.config.stiffness * (self.config.lower_limit - angle_rad)
        elif angle_rad >= self.config.upper_limit:
            tau += 10.0 * self.config.stiffness * (angle_rad - self.config.upper_limit)
        
        # 3. Damping
        tau -= self.config.damping * self.velocity
        
        # 4. Friction
        if abs(self.velocity) > 1e-6:
            tau -= self.config.friction * np.sign(self.velocity)
        
        # Clamp to max torque
        tau = np.clip(tau, -self.config.max_torque, self.config.max_torque)
        
        # Apply torque around axis to both bodies
        axis_world = self._get_axis_in_world_frame(body_a, self.axis)
        torque_a = tau * axis_world
        torque_b = -torque_a
        
        body_a.add_force(np.zeros(3), torque_a)  # add_force with position=0 applies torque
        body_b.add_force(np.zeros(3), torque_b)
    
    def _apply_spherical_constraint(self, body_a, body_b, dt):
        """
        SPHERICAL JOINT: 3 DOF full rotation around anchor point.
        
        Constraint: relative position at anchor must be zero (holds distance)
        
        Applied forces:
        1. Position constraint: F = -stiffness * position_error
        2. Damping: F = -damping * velocity
        """
        # Position of joint in world frames
        anchor_a = body_a.pos + self._get_anchor_in_world_frame(body_a)
        anchor_b = body_b.pos + self._get_anchor_in_world_frame(body_b)
        
        # Position error
        pos_error = anchor_b - anchor_a
        error_mag = np.linalg.norm(pos_error)
        
        if error_mag > 1e-6:
            direction = pos_error / error_mag
        else:
            direction = np.array([0, 0, 1])
        
        # Constraint force (prevents separation)
        force = -self.config.stiffness * error_mag * direction
        force = np.clip(force, -self.config.max_force, self.config.max_force)
        
        # Add damping (relative velocity along error direction)
        rel_vel = body_b.linear_vel - body_a.linear_vel
        vel_along_error = np.dot(rel_vel, direction)
        damp_force = -self.config.damping * vel_along_error * direction
        
        force += damp_force
        
        body_a.add_force(-force)
        body_b.add_force(force)
    
    def _apply_fixed_constraint(self, body_a, body_b, dt):
        """
        FIXED JOINT: 0 DOF rigid connection.
        
        Constraint: relative position and rotation = 0
        
        This is a strong constraint that keeps bodies rigidly connected.
        """
        # Position constraint
        anchor_a = body_a.pos + self._get_anchor_in_world_frame(body_a)
        anchor_b = body_b.pos + self._get_anchor_in_world_frame(body_b)
        pos_error = anchor_b - anchor_a
        
        # Very stiff spring to maintain position
        force = -100.0 * self.config.stiffness * pos_error
        force = np.clip(force, -self.config.max_force * 10, self.config.max_force * 10)
        
        body_a.add_force(-force)
        body_b.add_force(force)
        
        # Orientation constraint: align body orientations
        # Apply torque to reduce relative rotation
        relative_q = body_b.orientation * Quaternion.inverse(body_a.orientation)
        torque_mag = 4 * math.asin(math.sqrt(relative_q.x**2 + relative_q.y**2 + relative_q.z**2))
        
        if torque_mag > 1e-6:
            axis = np.array([relative_q.x, relative_q.y, relative_q.z]) / math.sin(torque_mag / 2)
            torque = -100.0 * self.config.stiffness * torque_mag * axis
        else:
            torque = np.zeros(3)
        
        torque = np.clip(torque, -self.config.max_torque * 10, self.config.max_torque * 10)
        
        body_a.add_force(np.zeros(3), -torque)
        body_b.add_force(np.zeros(3), torque)
    
    def _apply_hinge2_constraint(self, body_a, body_b, dt):
        """
        HINGE2 JOINT: 2 DOF rotation around two axes.
        
        Example: shoulder with pitch and yaw
        
        Constraints:
        - Rotation around axis1 and axis2 allowed
        - Rotation around other directions constrained
        """
        # Similar to revolute but with two independent rotational DOFs
        relative_quaternion = body_b.orientation * Quaternion.inverse(body_a.orientation)
        
        angle1 = self._extract_rotation_around_axis(relative_quaternion, self.axis)
        angle2 = self._extract_rotation_around_axis(relative_quaternion, self.axis2)
        
        # Clamp to limits
        angle1 = np.clip(angle1, self.config.lower_limit, self.config.upper_limit)
        angle2 = np.clip(angle2, self.config.lower_limit, self.config.upper_limit)
        
        # Apply torques for each axis (simplified)
        tau1 = -self.config.stiffness * angle1 - self.config.damping * self.velocity
        tau2 = -self.config.stiffness * angle2 - self.config.damping * self.velocity2
        
        axis1_world = self._get_axis_in_world_frame(body_a, self.axis)
        axis2_world = self._get_axis_in_world_frame(body_a, self.axis2)
        
        torque = tau1 * axis1_world + tau2 * axis2_world
        torque = np.clip(torque, -self.config.max_torque, self.config.max_torque)
        
        body_a.add_force(np.zeros(3), -torque)
        body_b.add_force(np.zeros(3), torque)
    
    def _apply_prismatic_constraint(self, body_a, body_b, dt):
        """
        PRISMATIC JOINT: 1 DOF linear motion along axis.
        
        Example: piston or sliding mechanism
        
        Constraint: relative position along axis within limits
        """
        # Compute relative position along axis
        rel_pos = body_b.pos - body_a.pos
        axis_normalized = self.axis.cpu().numpy() if isinstance(self.axis, torch.Tensor) else self.axis
        
        distance_along_axis = np.dot(rel_pos, axis_normalized)
        
        # Clamp to limits
        distance_clamped = np.clip(distance_along_axis, 
                                   self.config.lower_limit, 
                                   self.config.upper_limit)
        
        # Error in position
        position_error = distance_clamped - distance_along_axis
        
        # Constraint force along axis
        force_mag = -self.config.stiffness * position_error
        
        # Limit penalty
        if distance_along_axis <= self.config.lower_limit:
            force_mag -= 10.0 * self.config.stiffness * (self.config.lower_limit - distance_along_axis)
        elif distance_along_axis >= self.config.upper_limit:
            force_mag += 10.0 * self.config.stiffness * (distance_along_axis - self.config.upper_limit)
        
        # Damping
        rel_vel = body_b.linear_vel - body_a.linear_vel
        vel_along_axis = np.dot(rel_vel, axis_normalized)
        force_mag -= self.config.damping * vel_along_axis
        
        # Friction (static)
        if abs(vel_along_axis) < 0.01:
            force_mag = np.clip(force_mag, -self.config.friction, self.config.friction)
        else:
            force_mag -= self.config.friction * np.sign(vel_along_axis)
        
        # Clamp to max force
        force_mag = np.clip(force_mag, -self.config.max_force, self.config.max_force)
        
        force = force_mag * axis_normalized
        
        body_a.add_force(-force)
        body_b.add_force(force)
    
    # ===== Helper Methods =====
    
    def _extract_rotation_around_axis(self, quaternion, axis):
        """Extract rotation angle around a specific axis from quaternion."""
        # Convert quaternion to axis-angle representation
        angle = 2 * math.acos(np.clip(quaternion.w, -1, 1))
        
        if math.sin(angle / 2) > 1e-6:
            rotation_axis = np.array([quaternion.x, quaternion.y, quaternion.z]) / math.sin(angle / 2)
        else:
            return 0.0
        
        # Project onto desired axis
        if isinstance(axis, torch.Tensor):
            axis_np = axis.cpu().numpy()
        else:
            axis_np = axis
        
        rotation_along_axis = np.dot(rotation_axis, axis_np / np.linalg.norm(axis_np)) * angle
        return rotation_along_axis
    
    def _get_axis_in_world_frame(self, body, axis):
        """Rotate axis from body frame to world frame."""
        if isinstance(axis, torch.Tensor):
            axis_np = axis.cpu().numpy()
        else:
            axis_np = axis
        
        rotation_matrix = body.orientation.to_rotation_matrix()
        return rotation_matrix @ axis_np
    
    def _get_anchor_in_world_frame(self, body):
        """Rotate anchor position from body frame to world frame."""
        rotation_matrix = body.orientation.to_rotation_matrix()
        return rotation_matrix @ self.config.anchor_pos


class QuadrupedJointSetup:
    """Pre-configured joint setup for quadrupeds."""
    
    @staticmethod
    def create_quadruped_joints(device=None, dtype=None):
        """
        Create realistic joint constraints for quadruped.
        
        Each leg has:
        - Hip: Spherical joint (3 DOF abduction/adduction + movement)
        - Knee: Revolute joint (1 DOF bending)
        - Ankle: Revolute joint (1 DOF bending)
        """
        joints = []
        
        leg_configs = [
            ("FL", [0.15, 0.0, -0.1]),   # Front-left
            ("FR", [0.15, 0.0, 0.1]),    # Front-right
            ("BL", [-0.15, 0.0, -0.1]),  # Back-left
            ("BR", [-0.15, 0.0, 0.1]),   # Back-right
        ]
        
        for leg_name, hip_offset in leg_configs:
            # Hip joint: Spherical (3 DOF)
            hip_joint = JointConfig(
                joint_type=JointType.SPHERICAL,
                parent_link="torso",
                child_link=f"{leg_name}_hip",
                anchor_pos=np.array(hip_offset),
                axis=np.array([1, 0, 0]),  # Primary rotation axis (abduction)
                stiffness=0.2,
                damping=0.05,
                max_force=50.0
            )
            joints.append(JointConstraint(hip_joint, device=device, dtype=dtype))
            
            # Knee joint: Revolute (1 DOF pitch)
            knee_joint = JointConfig(
                joint_type=JointType.REVOLUTE,
                parent_link=f"{leg_name}_hip",
                child_link=f"{leg_name}_knee",
                anchor_pos=np.array([0, -0.1, 0]),
                axis=np.array([0, 1, 0]),  # Rotation around Y (pitch)
                lower_limit=-math.pi / 3,  # -60°
                upper_limit=0,              # 0° (can't bend backward)
                stiffness=0.5,
                damping=0.1,
                max_torque=20.0
            )
            joints.append(JointConstraint(knee_joint, device=device, dtype=dtype))
            
            # Ankle joint: Revolute (1 DOF pitch)
            ankle_joint = JointConfig(
                joint_type=JointType.REVOLUTE,
                parent_link=f"{leg_name}_knee",
                child_link=f"{leg_name}_foot",
                anchor_pos=np.array([0, -0.1, 0]),
                axis=np.array([0, 1, 0]),  # Rotation around Y (pitch)
                lower_limit=-math.pi / 6,  # -30°
                upper_limit=math.pi / 6,   # +30°
                stiffness=0.3,
                damping=0.08,
                max_torque=15.0
            )
            joints.append(JointConstraint(ankle_joint, device=device, dtype=dtype))
        
        return joints
