"""
Evaluation script: loads checkpoint and evaluates with rendering.
"""
import torch
from config import Config

if __name__ == "__main__":
    from source import System
    from source.renderer import Renderer
    
    try:
        app = System()
        print("System initialized successfully")
        print("Loading checkpoint for evaluation...")
        app.training.load_checkpoint()
        
        print("Starting evaluation phase with rendering...")
        renderer = Renderer(
            device=app.device,
            dtype=app.dtype,
            environment=app.env
        )
        
        app.training.model.eval()
        
        # Run a few evaluation episodes
        for eval_ep in range(Config.EVAL_EPISODES):
            print(f"\nEvaluation Episode {eval_ep + 1}/{Config.EVAL_EPISODES}")
            
            # Reset environment
            app.env.spawn_random_goal(0)
            creature = app.env.creatures[0]
            creature.pos = torch.tensor([0.0, 0.0, 0.5], dtype=app.dtype, device=app.device)
            creature.velocity = torch.zeros(3, dtype=app.dtype, device=app.device)
            creature.orientation = torch.zeros(3, dtype=app.dtype, device=app.device)
            creature.joint_angles = torch.tensor([0.3, 0.6, 0.3] * 4, dtype=app.dtype, device=app.device)
            creature.joint_velocities = torch.zeros(12, dtype=app.dtype, device=app.device)
            creature.foot_contact = torch.ones(4, dtype=app.dtype, device=app.device)
            
            h0, c0 = app.training.model.init_state(1, app.device, app.dtype)
            creature.rnn_state = (h0, c0)
            
            total_reward = 0.0
            goals_reached = 0
            
            for step in range(Config.MAX_STEPS_PER_EPISODE):
                # Render
                renderer.render(creature)
                
                # Get observation
                obs = app.env.observe(creature)
                
                # Get action from policy (deterministic - use mean)
                with torch.no_grad():
                    (mu, log_std), value, new_state = app.training.model(
                        obs, 
                        app.env.edge_indices[0],
                        prev_state=creature.rnn_state
                    )
                
                creature.rnn_state = new_state
                
                # Use mean action (no noise for evaluation)
                action = torch.tanh(mu)
                motor_torques = action[0] * 5.0
                
                # Apply physics
                reward, com_dist, _ = app.physics._compute_reward(
                    creature, motor_torques, app.env.goal_pos_t
                )
                
                total_reward += float(reward)
                
                # Check goal reached
                if float(com_dist) < app.physics.com_distance_threshold:
                    goals_reached += 1
                    app.env.spawn_random_goal(0)
                    
                    h0, c0 = app.training.model.init_state(1, app.device, app.dtype)
                    creature.rnn_state = (h0, c0)
            
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Goals Reached: {goals_reached}")
        
        renderer.close()
        print("\nEvaluation completed")
        
    except Exception as e:
        print(f"Fatal exception: {e}")
        import traceback
        traceback.print_exc()
    print("Application exited")
