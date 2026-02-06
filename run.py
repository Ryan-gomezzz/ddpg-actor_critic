"""
=============================================================================
RUN.PY - Evaluation Script for Trained DDPG/AC Models
=============================================================================
PURPOSE:
    Load a pre-trained model and evaluate it on the MountainCarContinuous-v0
    environment. Used AFTER training (train.py) to test performance.

KEY DIFFERENCE FROM TRAIN.PY:
    - Uses agent.predict() instead of agent.sample_act() (no exploration noise)
    - Loads saved weights instead of learning
    - Does NOT call save_transition() or learn()
    - Runs 100 test episodes (vs 1000 training episodes)
    - Tracks additional metrics: steps per episode, force usage

BEFORE RUNNING:
    Update the load_model() paths to match YOUR trained checkpoint.
    Paths point to result/{ddpg|ac}/{exp_name}/actor_epoch*_avgScore*.pth
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
# CONFIGURATION
# -----------------------------------------------------------------------------
exp_name = 'exp1'
agent_name = 'DDPG'
if_render = False  # True = visualize in window; False = headless (rgb_array for recording)

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # STEP 1: Create environment with optional rendering
    # -------------------------------------------------------------------------
    env = gym.make('MountainCarContinuous-v0',
                   render_mode='human' if if_render else 'rgb_array')
    env = env.unwrapped

    # -------------------------------------------------------------------------
    # STEP 2: Initialize agent and LOAD pre-trained weights
    # -------------------------------------------------------------------------
    # IMPORTANT: Update actor_path and critic_path to your trained model files
    if agent_name == 'AC':
        agent = ActorCriticAgent(env, hidden1=256, hidden2=256, gamma=0.99,
                                 lr_actor=0.00001, lr_critic=0.0001,
                                 exp_name=exp_name)
        agent.load_model(actor_path='result/ac/exp1/actor_epoch234_avgScore81.834.pth',
                         critic_path='result/ac/exp1/critic_epoch234_avgScore81.834.pth')
    elif agent_name == 'DDPG':
        agent = DDPGAgent(env, hidden1=256, hidden2=256, gamma=0.99,
                          lr_actor=0.0001, lr_critic=0.001,
                          exp_name=exp_name)
        agent.load_model(actor_path='./result/ddpg/exp1/actor_epoch878_avgScore90.220.pth',
                         critic_path='./result/ddpg/exp1/critic_epoch878_avgScore90.220.pth')

    # Print network architecture (useful for verification)
    print(agent.actor)
    print(agent.critic)

    # -------------------------------------------------------------------------
    # STEP 3: Run evaluation episodes (no exploration, deterministic/greedy)
    # -------------------------------------------------------------------------
    episodes = 100
    scores, avg_scores, turns, goals = [], [], [], []
    steps = []   # Steps per episode (efficiency metric)
    forces = []  # Average force magnitude (energy efficiency)

    for episode in tqdm.tqdm(range(episodes), file=sys.stdout):
        state = env.reset()[0]
        score = 0
        step = 0
        force = 0

        while True:
            env.render()  # Capture frame (or display if if_render=True)

            # predict() = greedy action, NO noise (unlike sample_act during training)
            action = agent.predict(state)

            # Track force magnitude for analysis (clipped to [0,1] per step)
            force += min(abs(action), 1)

            nxt_state, reward, done, _, info = env.step(action)
            score += reward
            step += 1
            state = nxt_state

            if done or _:
                break

        # Record metrics
        forces.append(force / step)
        steps.append(step)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        turns.append(episode)
        goals.append(90)

        print("Episode {0}/{1}, Score: {2:.3f}, AVG Score: {3:.3f}, STEP: {4}, AVG Force: {5}".format(
            episode, episodes, score, avg_score, step, force / step))

    # -------------------------------------------------------------------------
    # STEP 4: Plot test results and print summary
    # -------------------------------------------------------------------------
    # Saves: MountainCarContinuous-v0_{agent}_TEST.png in current directory
    utils.utils.plt_graph(turns, scores, avg_scores, goals,
                          'MountainCarContinuous-v0', agent_name, 'TEST')

    print('avg_score:{}, avg_step:{}, avg_force:{}'.format(
        np.mean(scores[-100:]), np.mean(steps[-100:]), np.mean(forces[-100:])))
