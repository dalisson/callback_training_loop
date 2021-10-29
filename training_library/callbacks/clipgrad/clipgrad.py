from ..callback import Callback
import torch


class ClipGradCallback(Callback):
    '''
    A callback to train on GPU
    '''
    order = 1

    def __init__(self, max_grad):
        super(ClipGradCallback, self).__init__()
        self.m_g = max_grad

    def after_loss_backward(self):
        torch.nn.utils.clip_grad_norm_(self.run.model.parameters(), self.m_g)
