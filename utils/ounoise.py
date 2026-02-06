"""
=============================================================================
OUNOISE - Ornstein-Uhlenbeck Process for Exploration
=============================================================================
Generates temporally correlated noise for exploration in continuous action
spaces. OU noise is preferred over white noise for physical control tasks
because it produces smoother, more realistic exploration trajectories.

FORMULA:
    dx = theta * (mu - x) + sigma * N(0,1)
    x_{t+1} = x_t + dx

WHY OU NOISE?
    MountainCarContinuous is difficult; the car must learn to swing back
    and forth. Deterministic policy without noise rarely discovers this.
    OU noise is added to actor output during training (sample_act).
=============================================================================
"""

import copy

import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for correlated exploration noise.
    Produces noise that tends toward mu with mean-reversion strength theta.
    """

    def __init__(self, size, mu, theta, sigma):
        """
        :param size:  Action dimension (e.g., 1 for MountainCarContinuous)
        :param mu:    Mean of the noise (e.g., 0.2)
        :param theta: Mean-reversion rate (how fast noise returns to mu)
        :param sigma: Volatility (std of random component)
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset internal state to mean (mu). Call at episode start if desired."""
        self.state = copy.deepcopy(self.mu)

    def sample(self):
        """
        Update internal state and return noise sample.
        OU process: dx = theta*(mu - x) + sigma*randn; x = x + dx
        :return: noise array of shape (size,)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
