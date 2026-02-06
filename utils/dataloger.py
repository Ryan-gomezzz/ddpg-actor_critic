"""
=============================================================================
DATALOGER - TensorBoard Logging Utility
=============================================================================
Wrapper around PyTorch's SummaryWriter for logging training metrics.
Used by DDPGAgent and ActorCriticAgent to log actor_loss and critic_loss.

USAGE:
    loger = DataLoger('./result/ddpg/exp1')
    loger.log('actor_loss', 0.5, step=1000)

VIEW LOGS:
    tensorboard --logdir=./result/ddpg/exp1
=============================================================================
"""

from torch.utils.tensorboard import SummaryWriter


class DataLoger():
    """
    Simple wrapper for TensorBoard scalar logging.
    Logs are saved to the given directory for later visualization.
    """

    def __init__(self, dir):
        """
        :param dir: Directory path for TensorBoard event files (e.g., ./result/ddpg/exp1)
        """
        self.writer = SummaryWriter(dir)

    def log(self, tag, value, step):
        """
        Log a scalar value to TensorBoard.
        :param tag:  Name of the metric (e.g., 'actor_loss', 'critic_loss')
        :param value: Scalar value to log
        :param step:  Global step (x-axis in TensorBoard)
        """
        self.writer.add_scalar(tag, value, step)
