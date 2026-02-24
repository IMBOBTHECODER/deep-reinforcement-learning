"""
Contact and friction models for physics engine.
Separated from main physics engine for clarity.
"""

import numpy as np


class ContactModel:
    """Handles all contact detection and force computation."""
    
    def __init__(self, config):
        self.ground_level = config.GROUND_LEVEL
        self.foot_height_threshold = config.FOOT_HEIGHT_THRESHOLD
        self.contact_stiffness = config.CONTACT_STIFFNESS
        self.contact_damping = config.CONTACT_DAMPING
        self.contact_restitution = config.CONTACT_RESTITUTION
        
        # Friction
        self.friction_model = getattr(config, 'FRICTION_MODEL', 'coulomb+viscous')
        self.friction_coeff_static = getattr(config, 'FRICTION_COEFFICIENT_STATIC', 0.9)
        self.friction_coeff_kinetic = getattr(config, 'FRICTION_COEFFICIENT_KINETIC', 0.85)
        self.friction_viscous_damping = getattr(config, 'FRICTION_VISCOUS_DAMPING', 0.05)
        self.friction_slip_threshold = getattr(config, 'FRICTION_SLIP_VELOCITY_THRESHOLD', 0.01)
        
        # Friction cones
        self.use_friction_cones = getattr(config, 'USE_FRICTION_CONES', True)
        self.friction_cone_damping = getattr(config, 'FRICTION_CONE_DAMPING', 0.3)
    
    def detect_contacts(self, foot_positions):
        """
        Detect which feet are in contact with ground.
        
        Args:
            foot_positions: (4, 3) array of foot positions
        
        Returns:
            contact_mask: (4,) boolean array
            penetrations: (4,) array of penetration depths
        """
        contact_mask = np.zeros(4, dtype=bool)
        penetrations = np.zeros(4)
        
        for foot_idx in range(4):
            foot_z = float(foot_positions[foot_idx, 2])
            if foot_z <= self.ground_level + self.foot_height_threshold:
                contact_mask[foot_idx] = True
                penetrations[foot_idx] = max(0, self.ground_level - foot_z)
        
        return contact_mask, penetrations
    
    def compute_contact_forces(self, foot_positions, body_linear_vel):
        """
        Compute spring-damper contact forces with restitution.
        
        Args:
            foot_positions: (4, 3) array of foot positions
            body_linear_vel: (3,) body velocity
        
        Returns:
            contact_force_z: scalar vertical force
            num_contacts: integer count of contacting feet
        """
        contact_mask, penetrations = self.detect_contacts(foot_positions)
        num_contacts = np.sum(contact_mask)
        contact_force_z = 0.0
        
        for foot_idx in range(4):
            if contact_mask[foot_idx]:
                penetration = penetrations[foot_idx]
                
                # Spring force
                spring_force = self.contact_stiffness * penetration
                
                # Damper force
                damper_force = self.contact_damping * float(body_linear_vel[2])
                
                # Restitution force
                restitution_force = (self.contact_restitution * 
                                    (-float(body_linear_vel[2])) * 
                                    (self.contact_stiffness * penetration / max(penetration, 0.001)))
                
                # Net contact force (unilateral constraint: cannot pull)
                foot_force = max(0.0, spring_force - damper_force + restitution_force)
                contact_force_z += foot_force
        
        if num_contacts > 0:
            contact_force_z /= num_contacts
        
        return contact_force_z, num_contacts
    
    def compute_friction_force(self, normal_force, slip_velocity):
        """
        Compute friction force using configured model.
        
        Args:
            normal_force: scalar normal contact force
            slip_velocity: scalar horizontal slip speed
        
        Returns:
            friction_force: scalar magnitude of friction force
        """
        if slip_velocity < self.friction_slip_threshold:
            # Low slip: static friction
            coulomb_part = self.friction_coeff_static * normal_force
        else:
            # Active slip: kinetic friction
            coulomb_part = self.friction_coeff_kinetic * normal_force
        
        # Viscous damping
        viscous_part = self.friction_viscous_damping * slip_velocity
        
        return coulomb_part + viscous_part
    
    def apply_friction(self, contact_force_z, body_linear_vel, num_contacts):
        """
        Apply friction forces with optional friction cone constraint.
        
        Args:
            contact_force_z: vertical contact force
            body_linear_vel: (3,) body velocity
            num_contacts: number of contacting feet
        
        Returns:
            friction_force_xy: (2,) horizontal friction force
        """
        if num_contacts == 0 or contact_force_z <= 0:
            return np.zeros(2)
        
        normal_force = contact_force_z
        foot_vel_horizontal = np.sqrt(body_linear_vel[0]**2 + body_linear_vel[1]**2)
        
        # Compute friction magnitude
        friction_force = self.compute_friction_force(normal_force, foot_vel_horizontal)
        
        if foot_vel_horizontal <= self.friction_slip_threshold:
            return np.zeros(2)
        
        # Direction opposite to velocity
        direction = np.array([body_linear_vel[0], body_linear_vel[1]]) / (foot_vel_horizontal + 1e-8)
        friction_vector = -friction_force * direction / num_contacts
        
        # Apply friction cone constraint if enabled
        if self.use_friction_cones:
            friction_magnitude = np.linalg.norm(friction_vector)
            max_friction = self.friction_coeff_kinetic * normal_force / num_contacts
            if friction_magnitude > max_friction:
                friction_vector = friction_vector * (max_friction / (friction_magnitude + 1e-8))
            
            # Add cone damping
            damping_vector = -self.friction_cone_damping * direction / num_contacts
            friction_vector += damping_vector
        
        return friction_vector[:2]