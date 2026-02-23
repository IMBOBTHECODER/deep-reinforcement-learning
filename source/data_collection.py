"""
Data collection for PPO training.
Separates trajectory collection logic from main simulator.
"""

import torch
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def collect_trajectories_vectorized(training_engine, environment, physics_engine, max_steps=500):
    """
    Collect trajectories from multiple parallel environments.
    
    Args:
        training_engine: TrainingEngine with RL model
        environment: Environment instance
        physics_engine: PhysicsEngine instance
        max_steps: Max steps per episode
    
    Returns:
        list of trajectories, one per environment
    """
    training_engine.model.eval()
    
    num_envs = environment.num_envs
    trajectories = [[] for _ in range(num_envs)]
    step_counts = [0] * num_envs
    
    # Reset all creatures
    for env_id in range(num_envs):
        creature = environment.creatures[env_id]
        creature.pos = torch.tensor([0.0, 0.0, 0.5], dtype=training_engine.dtype, device=training_engine.device)
        creature.velocity = torch.zeros(3, dtype=training_engine.dtype, device=training_engine.device)
        creature.orientation = torch.zeros(3, dtype=training_engine.dtype, device=training_engine.device)
        creature.joint_angles = torch.tensor([0.3, 0.6, 0.3] * 4, dtype=training_engine.dtype, device=training_engine.device)
        creature.joint_velocities = torch.zeros(12, dtype=training_engine.dtype, device=training_engine.device)
        creature.foot_contact = torch.ones(4, dtype=training_engine.dtype, device=training_engine.device)
        h0, c0 = training_engine.model.init_state(1, training_engine.device, training_engine.dtype)
        creature.rnn_state = (h0, c0)
        environment.spawn_random_goal(env_id)
    
    # Main collection loop: run all environments until completion
    LOG_2PI = 2.3978953  # math.log(2.0 * math.pi)
    
    for global_step in range(max_steps):
        obs_list = []
        for env_id in range(num_envs):
            if step_counts[env_id] < max_steps:
                creature = environment.creatures[env_id]
                obs = environment.observe(creature)
                obs_list.append(obs)
        
        if not obs_list:
            break
        
        obs_batch = torch.cat(obs_list, dim=0)
        
        # Get actions from policy
        with torch.no_grad():
            edge_idx = environment.edge_indices[0]
            
            h_list = []
            c_list = []
            for env_id in range(num_envs):
                if step_counts[env_id] < max_steps:
                    creature = environment.creatures[env_id]
                    h, c = creature.rnn_state
                    h_list.append(h)
                    c_list.append(c)
            
            h_batch = torch.cat(h_list, dim=0)
            c_batch = torch.cat(c_list, dim=0)
            
            (mu, log_std), values, (new_h, new_c) = training_engine.model(
                obs_batch, edge_idx, prev_state=(h_batch, c_batch)
            )
            
            std = torch.exp(log_std)
            actions = mu + torch.randn_like(mu) * std
            
            log_probs = -0.5 * ((actions - mu) ** 2 / (std ** 2)).sum(dim=1)
            log_probs = log_probs - log_std.sum(dim=1) - 0.5 * 12 * LOG_2PI
            
            # Collect for reward computation
            active_creatures = []
            active_actions = []
            active_env_ids = []
            env_id_active = 0
            
            for env_id in range(num_envs):
                if step_counts[env_id] < max_steps:
                    creature = environment.creatures[env_id]
                    active_creatures.append(creature)
                    active_actions.append(actions[env_id_active])
                    active_env_ids.append(env_id)
                    env_id_active += 1
            
            # Compute rewards in parallel
            from source.simulate import compute_reward_parallel
            reward_results = compute_reward_parallel(
                active_creatures, active_actions, physics_engine, environment.goal_pos_t
            )
            
            # Store trajectory data
            env_id_active = 0
            for env_idx, (creature, env_id) in enumerate(zip(active_creatures, active_env_ids)):
                if step_counts[env_id] < max_steps:
                    mu_i = mu[env_id_active:env_id_active+1]
                    action_i = actions[env_id_active:env_id_active+1]
                    value_i = values[env_id_active:env_id_active+1]
                    log_prob_i = log_probs[env_id_active]
                    
                    creature.rnn_state = (
                        new_h[env_id_active:env_id_active+1],
                        new_c[env_id_active:env_id_active+1]
                    )
                    
                    reward, com_dist = reward_results[env_idx]
                    
                    done = float(com_dist) < physics_engine.com_distance_threshold
                    if done:
                        environment.spawn_random_goal(env_id)
                    
                    trajectories[env_id].append({
                        'obs': obs_list[env_idx].squeeze(0),
                        'action': action_i.squeeze(0),
                        'value': value_i.squeeze(0),
                        'log_prob': log_prob_i,
                        'reward': reward,
                        'done': torch.tensor(done, dtype=training_engine.dtype, device=training_engine.device),
                    })
                    
                    step_counts[env_id] += 1
                    env_id_active += 1
    
    return trajectories
