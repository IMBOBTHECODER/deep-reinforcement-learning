import math
import pyray as pr
import torch
import logging

from config import Config

logger = logging.getLogger(__name__)


class Renderer:
    """Handles 3D visualization with PyRay."""
    
    def __init__(self, device, dtype, environment):
        self.device = device
        self.dtype = dtype
        self.env = environment
        
        # Window dimensions
        self.window_w = Config.DIM[0] * Config.SCALE
        self.window_h = Config.DIM[1] * Config.SCALE
        
        # Entity rendering data
        self.entity_colors = {
            en_id: data["color"]
            for en_id, data in Config.ENTITY_TYPES.items()
        }
        self.entity_sizes = {
            en_id: data["size"]
            for en_id, data in Config.ENTITY_TYPES.items()
        }
        
        # Window state
        self.window_initialized = False
        self.camera = None
        self.camera_speed = 2.0
        self.step_count = 0
    
    def initialize_window(self):
        """Initialize PyRay window and camera (called on first render)."""
        if self.window_initialized:
            return
        
        pr.init_window(self.window_w, self.window_h, "Reinforcement Learning 3D")
        logger.info("Window created for evaluation")
        pr.set_target_fps(Config.TARGET_FPS)
        
        # Setup 3D camera
        self.camera = pr.Camera3D()
        self.camera.position = pr.Vector3(self.env.w * 0.8, -self.env.h * 0.5, self.env.d * 0.8)
        self.camera.target = pr.Vector3(self.env.w / 2, self.env.h / 2, self.env.d / 2)
        self.camera.up = pr.Vector3(0, 1, 0)
        self.camera.fovy = 45.0
        self.camera.projection = 0
        
        self.window_initialized = True
    
    def close(self):
        """Close rendering window."""
        if self.window_initialized:
            pr.close_window()
    
    def render(self):
        """Render 3D environment with PyRay."""
        self.handle_camera_controls()
        
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        
        # 3D scene
        pr.begin_mode_3d(self.camera)
        
        # Floor
        pr.draw_plane(pr.Vector3(self.env.w/2, 0, self.env.d/2), pr.Vector2(self.env.w, self.env.d), pr.DARKGRAY)
        
        # Grid
        for x in range(0, self.env.w, 10):
            pr.draw_line_3d(pr.Vector3(x, 0, 0), pr.Vector3(x, 0, self.env.d), pr.GRAY)
        for z in range(0, self.env.d, 10):
            pr.draw_line_3d(pr.Vector3(0, 0, z), pr.Vector3(self.env.w, 0, z), pr.GRAY)
        
        # Creatures
        for creature in self.env.creatures:
            en_id = creature.en_id
            pos = (float(creature.pos[0]), float(creature.pos[1]), float(creature.pos[2]))
            size = self.entity_sizes[en_id]
            color_rgb = self.entity_colors[en_id]
            rgb_color = pr.Color(color_rgb[0], color_rgb[1], color_rgb[2], 255)
            
            cube_pos = pr.Vector3(pos[0], size[1]/2, pos[2])
            cube_size = pr.Vector3(float(size[0]), float(size[1]), float(size[2]))
            
            pr.draw_cube(cube_pos, cube_size.x, cube_size.y, cube_size.z, rgb_color)
            pr.draw_cube_wires(cube_pos, cube_size.x, cube_size.y, cube_size.z, pr.WHITE)
        
        # Goal
        size = self.entity_sizes[2]
        color_rgb = self.entity_colors[2]
        rgb_color = pr.Color(color_rgb[0], color_rgb[1], color_rgb[2], 255)
        cube_pos = pr.Vector3(float(self.env.goal_pos_t[0]), size[1]/2, float(self.env.goal_pos_t[2]))
        cube_size = pr.Vector3(float(size[0]), float(size[1]), float(size[2]))
        pr.draw_cube(cube_pos, cube_size.x, cube_size.y, cube_size.z, rgb_color)
        pr.draw_cube_wires(cube_pos, cube_size.x, cube_size.y, cube_size.z, pr.WHITE)
        
        pr.end_mode_3d()
        
        # 2D overlay
        pr.draw_fps(10, 10)
        
        x, y, z = float(self.env.creatures[0].pos[0]), float(self.env.creatures[0].pos[1]), float(self.env.creatures[0].pos[2])
        pr.draw_text(f"Agent: ({x:.1f}, {y:.1f}, {z:.1f})", 10, 30, 20, pr.WHITE)
        
        gx, gy, gz = float(self.env.goal_pos_t[0]), float(self.env.goal_pos_t[1]), float(self.env.goal_pos_t[2])
        pr.draw_text(f"Goal: ({gx:.0f}, {gy:.0f}, {gz:.0f})", 10, 50, 16, pr.YELLOW)
        
        dist = self.env._distance_to_goal(self.env.creatures[0])
        dist_float = float(dist)
        pr.draw_text(f"Distance: {dist_float:.1f}", 10, 70, 16, pr.WHITE)
        
        pr.draw_text(f"Steps: {self.step_count}", 10, 90, 16, pr.BLUE)
        pr.draw_text("WASD=Pan  LMB=Rotate", 10, 130, 14, pr.DARKGRAY)
        
        pr.end_drawing()
    
    def _normalize_vector(self, v):
        """Normalize a 3-tuple vector."""
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        if norm > 0:
            return (v[0] / norm, v[1] / norm, v[2] / norm)
        return v
    
    def _move_camera(self, direction):
        """Move camera in direction by camera_speed."""
        dx, dy, dz = self._normalize_vector(direction)
        self.camera.position.x += dx * self.camera_speed
        self.camera.position.y += dy * self.camera_speed
        self.camera.position.z += dz * self.camera_speed
        self.camera.target.x += dx * self.camera_speed
        self.camera.target.y += dy * self.camera_speed
        self.camera.target.z += dz * self.camera_speed
    
    def handle_camera_controls(self):
        """Handle interactive camera controls."""
        # Forward/Backward
        if pr.is_key_down(pr.KEY_W):
            dir_vec = (
                self.camera.target.x - self.camera.position.x,
                self.camera.target.y - self.camera.position.y,
                self.camera.target.z - self.camera.position.z
            )
            self._move_camera(dir_vec)
        
        if pr.is_key_down(pr.KEY_S):
            dir_vec = (
                self.camera.target.x - self.camera.position.x,
                self.camera.target.y - self.camera.position.y,
                self.camera.target.z - self.camera.position.z
            )
            self._move_camera((-dir_vec[0], -dir_vec[1], -dir_vec[2]))
        
        # Strafe
        if pr.is_key_down(pr.KEY_A) or pr.is_key_down(pr.KEY_D):
            forward = (
                self.camera.target.x - self.camera.position.x,
                self.camera.target.y - self.camera.position.y,
                self.camera.target.z - self.camera.position.z
            )
            up = (self.camera.up.x, self.camera.up.y, self.camera.up.z)
            
            right = (
                forward[1] * up[2] - forward[2] * up[1],
                forward[2] * up[0] - forward[0] * up[2],
                forward[0] * up[1] - forward[1] * up[0]
            )
            
            if pr.is_key_down(pr.KEY_A):
                self._move_camera((-right[0], -right[1], -right[2]))
            if pr.is_key_down(pr.KEY_D):
                self._move_camera(right)
        
        # Mouse rotation
        if pr.is_mouse_button_down(0):
            delta = pr.get_mouse_delta()
            angle_yaw = delta.x * 0.005
            angle_pitch = delta.y * 0.005
            
            dx = self.camera.position.x - self.camera.target.x
            dy = self.camera.position.y - self.camera.target.y
            dz = self.camera.position.z - self.camera.target.z
            
            cos_yaw, sin_yaw = math.cos(angle_yaw), math.sin(angle_yaw)
            dx_new = dx * cos_yaw - dz * sin_yaw
            dz_new = dx * sin_yaw + dz * cos_yaw
            
            horiz_dist = math.sqrt(dx_new**2 + dz_new**2)
            cos_pitch, sin_pitch = math.cos(angle_pitch), math.sin(angle_pitch)
            
            dy_new = dy * cos_pitch - horiz_dist * sin_pitch
            horiz_dist_new = dy * sin_pitch + horiz_dist * cos_pitch
            
            if horiz_dist > 1e-6:
                scale = horiz_dist_new / horiz_dist
                dx_new *= scale
                dz_new *= scale
            
            self.camera.position.x = self.camera.target.x + dx_new
            self.camera.position.y = self.camera.target.y + dy_new
            self.camera.position.z = self.camera.target.z + dz_new
