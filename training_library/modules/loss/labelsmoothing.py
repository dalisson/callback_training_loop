'''
Label smoothing loss, a regularization technique to make the model less
certain of its prediction, specially useful when the datase is not
fully curated
'''

import torch.nn as nn
from torch.nn.functional import nll_loss

def reduce_loss(loss, reduction='mean'):
    '''
    reduction method for the loss
    '''
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def lin_comb(v1, v2, beta):
    '''
    Simple linear combination
    '''
    return beta*v1 + (1-beta)*v2

class LabelSmoothingNLLLoss(nn.Module):
    '''
    Cross entropy with label smoothing
    '''
    def __init__(self, e: float = 0.1, reduction='mean'):
        super().__init__()
        self.e, self.reduction = e, reduction

    def forward(self, *inputs):
        c = self.inputs[0].size()[-1]
        loss = reduce_loss(-inputs[0].sum(dim=-1), reduction=self.reduction)
        nll = nll_loss(inputs[0], inputs[1], reduction=self.reduction)
        return lin_comb(loss/c, nll, self.e)
