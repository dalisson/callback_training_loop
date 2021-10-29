'''
This callback is a data agumentation technique called mixup
'''
from functools import partial
import torch
from torch import tensor
from torch.distributions.beta import Beta
from ..callback import Callback
from ...utils import listfy


def lin_comb(v1, v2, beta):
    '''
    Simple linear combination
    '''
    return beta*v1 + (1-beta)*v2

#context manager
class NoneReduce():
    '''
    A small context manager to keep the loss function from
    reducing the loss on mixup
    '''
    def __init__(self, loss_func):
        self.loss_func, self.old_red = loss_func, None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, 'reduction', self.old_red)

def unsqueeze(inp, dims):
    '''
    helper function for mixup callback
    '''
    for dim in listfy(dims):
        inp = torch.unsqueeze(inp, dim)
    return inp

def reduce_loss(loss, reduction='mean'):
    '''
    reduction method for the loss
    '''
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class MixUpCallback(Callback):
    order = 90 #Runs after normalization and cuda
    def __init__(self, α: float = 0.4):
        super().__init__()
        self.distrib = Beta(tensor([α]), tensor([α]))
        self.old_loss_func = None
        self.λ, self.yb1 = None, None

    def begin_fit(self):
        self.old_loss_func, self.run.loss_func = self.run.loss_func, self.loss_func

    def begin_batch(self):
        '''
        Mix the x_batch and y_batch at the beginning of each batch
        '''
        if not self.in_train:
            return #Only mixup things during training
        λ = self.distrib.sample((self.y_batch.size(0),)).squeeze().to(self.x_batch.device)
        λ = torch.stack([λ, 1-λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], (1, 2, 3))
        shuffle = torch.randperm(self.y_batch.size(0)).to(self.x_batch.device)
        xb1, self.yb1 = self.x_batch[shuffle], self.y_batch[shuffle]
        self.run.xb = lin_comb(self.x_batch, xb1, self.λ)

    def after_fit(self):
        '''
        Returns the loss function to the original loss function
        '''
        self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        '''
        The loss function for the mixup
        '''
        if not self.in_train:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))
