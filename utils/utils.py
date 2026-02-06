"""
=============================================================================
UTILS - Plotting and Helper Functions
=============================================================================
Provides plt_graph() for visualizing training/evaluation curves.
Used by train.py (TRAIN plot) and run.py (TEST plot).

OUTPUT:
    Saves PNG: {save_path}/{env_name}_{model_name}_{exp_name}.png
    Example: ./result/DDPG/exp2/MountainCarContinuous-v0_DDPG_TRAIN.png
=============================================================================
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def plt_graph(episodes, scores, avg_scores, goals, env_name, model_name, exp_name, save_path='./'):
    """
    Plot training/evaluation curves: score, rolling average, and target line.

    :param episodes:   Episode indices (x-axis)
    :param scores:     Raw score per episode (blue line)
    :param avg_scores: Rolling 100-episode average (orange dashed)
    :param goals:      Target score line (red dashed) - typically 90 for "solved"
    :param env_name:   Environment name (e.g., MountainCarContinuous-v0)
    :param model_name: Agent name (DDPG or AC)
    :param exp_name:   Experiment type (TRAIN or TEST)
    :param save_path:  Directory to save plot (default current dir)
    """
    df = pd.DataFrame({
        'x': episodes,
        'Score': scores,
        'Average Score': avg_scores,
        'Solved Requirement': goals
    })

    plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
    plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2,
             linestyle='dashed', label='AverageScore')
    plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2,
             linestyle='dashed', label='Solved Requirement')
    plt.legend()
    plt.ylim(70, 110)  # Y-axis range for MountainCar score scale
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path + '/' + '{}_{}_{}.png'.format(env_name, model_name, exp_name))
    plt.close()
