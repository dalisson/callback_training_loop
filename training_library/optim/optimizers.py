'''
Builds the optimizers
'''
from .optim import StatefulOptimizer
from .stats import AverageGradStat, AverageSqrGradStat, StepCountStat
from .steppers import weight_decay_step, sgd_with_momentum_step, adam_step,\
                      lamb_step
from ..utils import listfy

__all__ = ['sgd', 'adam', 'lamb']

def set_optim(parameters, optim_type, lr, **kwargs):
    assert optim_type in ['sgd', 'adam', 'lamb'], 'Optimizer Not Implemented'
    lrs = listfy(lr)
    if optim_type == 'sgd':
        steps = [sgd_with_momentum_step, weight_decay_step]
        stats = [AverageGradStat(dampening=False)]
    elif optim_type == 'adam':
        steps = [adam_step, weight_decay_step]
        stats = [StepCountStat(), AverageGradStat(), AverageSqrGradStat()]
    elif optim_type == 'lamb':
        steps = [lamb_step, weight_decay_step]
        stats = [StepCountStat(), AverageGradStat(), AverageSqrGradStat()]

    optim = StatefulOptimizer(parameters, steps, stats, lr=lrs[0], **kwargs)
    set_lr_for_groups(optim, lrs)
    return optim

def set_lr_for_groups(optim, lrs):
    n_groups = len(optim.param_groups)
    if len(lrs) != n_groups:
        min_lr, max_lr = min(lrs), max(lrs)
        step = (max_lr-min_lr)/(n_groups - 1)
        lrs = [(min_lr + x*step) for x in range(n_groups)]
    for group, lr in zip(optim.param_groups, lrs):
        group['lr'] = lr

def sgd(parameters, lr, momentum=0, weight_decay=1e-5):
    '''
    Builds a sgd optimizer
    '''

    return set_optim(parameters, 'sgd', lr=lr, momentum=momentum, wd=weight_decay)


def adam(parameters, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5, eps=1e-6):
    '''
    Builds the adam optimizer
    '''
    assert isinstance(betas, (tuple, list)) and len(betas) == 2
    return set_optim(parameters, 'adam', lr=lr, momentum=betas[0],
                     sqr_mom=betas[1], wd=weight_decay, eps=eps)

def lamb(parameters, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5, eps=1e-6):
    '''
    Builds the adam optimizer
    '''
    assert isinstance(betas, (tuple, list)) and len(betas) == 2
    return set_optim(parameters, 'lamb', lr=lr, momentum=betas[0],
                     sqr_mom=betas[1], wd=weight_decay, eps=eps)
