"""
=============================================================================
TRAIN.PY - Main Training Script for DDPG/AC Reinforcement Learning
=============================================================================
PURPOSE:
    This is the primary entry point for training the reinforcement learning
    agent. It runs the training loop, collects experience, and saves models
    and plots to the result/ directory.

FLOW:
    1. Create MountainCarContinuous-v0 environment
    2. Initialize agent (DDPG or AC based on agent_name)
    3. For each episode: interact with env -> save transitions -> learn
    4. Save model when avg_score >= 80 (solved threshold)
    5. Plot training curves and save to result/

DEPENDENCIES:
    - ddpg.ddpg_agent.DDPGAgent OR ac.ac_agent.ActorCriticAgent
    - utils.utils (for plt_graph)
    - gym (OpenAI Gym environment)
=============================================================================
"""

import sys

import torch
import tqdm

import utils.utils
from ac.ac_agent import ActorCriticAgent
from ddpg.ddpg_agent import DDPGAgent
import gym
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION: Change these to switch agent type or experiment
# -----------------------------------------------------------------------------
exp_name = 'exp2'      # Experiment name; results saved to result/{agent}/{exp_name}/
agent_name = 'DDPG'    # 'DDPG' or 'AC' - which algorithm to train

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # STEP 1: Create the RL environment
    # -------------------------------------------------------------------------
    # MountainCarContinuous-v0: Car in valley must reach hilltop flag
    # State: [position, velocity] (2 dims) | Action: force [-1, 1] (1 dim)
    env = gym.make('MountainCarContinuous-v0')
    env = env.unwrapped  # Remove step/time limits for full control

    # -------------------------------------------------------------------------
    # STEP 2: Initialize the RL agent (DDPG or Actor-Critic)
    # -------------------------------------------------------------------------
    if agent_name == 'AC':
        agent = ActorCriticAgent(env, hidden1=256, hidden2=256, gamma=0.99,
                                 lr_actor=0.00001, lr_critic=0.0001,
                                 exp_name=exp_name)
    elif agent_name == 'DDPG':
        agent = DDPGAgent(env, hidden1=256, hidden2=256, gamma=0.99,
                          lr_actor=0.0001, lr_critic=0.001,
                          exp_name=exp_name)

    # -------------------------------------------------------------------------
    # STEP 3: Training loop - run episodes and learn from experience
    # -------------------------------------------------------------------------
    episodes = 1000
    scores = []       # Raw score per episode
    avg_scores = []   # Rolling 100-episode average (smooths learning curve)
    turns = []        # Episode indices for plotting
    goals = []        # Target score line (90) for "solved" threshold
    step = 0          # Global step counter (used for buffer, learning, logging)

    for episode in tqdm.tqdm(range(episodes), file=sys.stdout):
        state = env.reset()[0]  # [0] extracts observation from (obs, info) tuple
        score = 0

        # Single episode: interact until done or truncated
        while True:
            step += 1

            # Sample action with exploration (noise for DDPG, stochastic for AC)
            action = agent.sample_act(state)

            # Execute action in environment
            nxt_state, reward, done, _, info = env.step(action)
            score += reward

            # Store transition (s, a, r, s', done) in replay buffer for off-policy learning
            agent.save_transition(state, action, reward, nxt_state, done)

            # Update actor and critic networks using batch from replay buffer
            agent.learn(episode, step, state)

            state = nxt_state
            if done or _:  # done=terminated, _=truncated (e.g. max steps)
                break

        # Record metrics for plotting
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # 100-episode moving average
        avg_scores.append(avg_score)
        turns.append(episode)
        goals.append(90)  # "Solved" threshold line on plot

        print("Episode {0}/{1}, Score: {2}, AVG Score: {3}".format(
            episode, episodes, score, avg_score))

        # Save checkpoint when agent reaches good performance (avg score >= 80)
        if avg_score >= 80:
            agent.save_model(episode, avg_score)

    # -------------------------------------------------------------------------
    # STEP 4: Plot and save training curves
    # -------------------------------------------------------------------------
    # Saves: result/{agent_name}/{exp_name}/MountainCarContinuous-v0_{agent}_TRAIN.png
    utils.utils.plt_graph(turns, scores, avg_scores, goals,
                          'MountainCarContinuous-v0', agent_name, 'TRAIN',
                          './result/{}/{}'.format(agent_name, exp_name))
