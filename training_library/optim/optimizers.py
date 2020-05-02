'''
Builds the optimizers
'''
from .optim import StatefulOptimizer
from .stats import AverageGradStat, AverageSqrGradStat, StepCountStat
from .steppers import weight_decay_step, sgd_with_momentum_step, adam_step,\
                      lamb_step
from ..utils import listfy

__all__ = ['sgd', 'adam', 'lamb']

def get_param_groups(net):
    groups = []
    for child in net.children():
        groups.append(list(child.parameters()))
    return groups

def set_lr_for_groups(optim, lrs):
    n_groups = len(optim.param_groups)
    if len(lrs) != n_groups:
        min_lr, max_lr = min(lrs), max(lrs)
        step = (max_lr-min_lr)/(n_groups - 1)
        lrs = [(min_lr + x*step) for x in range(n_groups)]
    for group, lr in zip(optim.param_groups, lrs):
        group['lr'] = lr

def sgd(model, lr, mom=0, weight_decay=1e-5):
    '''
    Builds a sgd optimizer
    '''
    lrs = listfy(lr)
    optim = StatefulOptimizer(get_param_groups(model), [weight_decay_step, sgd_with_momentum_step],
                              [AverageGradStat(dampening=False)], lr=lrs[0], mom=mom,
                              wd=weight_decay)
    set_lr_for_groups(optim, lrs)
    return optim

def adam(model, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5):
    '''
    Builds the adam optimizer
    '''
    lrs = listfy(lr)
    assert isinstance(betas, (tuple, list)) and len(betas) == 2
    optim = StatefulOptimizer(get_param_groups(model), [weight_decay_step, adam_step],
                              [StepCountStat(), AverageGradStat(), AverageSqrGradStat()],
                              lr=lrs[0], mom=betas[0], sqr_mom=betas[1], wd=weight_decay)
    set_lr_for_groups(optim, lrs)
    return optim

def lamb(model, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5):
    '''
    Builds the adam optimizer
    '''
    lrs = listfy(lr)
    assert isinstance(betas, (tuple, list)) and len(betas) == 2
    optim = StatefulOptimizer(get_param_groups(model), [weight_decay_step, lamb_step],
                              [StepCountStat(), AverageGradStat(), AverageSqrGradStat()],
                              lr=lrs[0], mom=betas[0], sqr_mom=betas[1], wd=weight_decay)
    set_lr_for_groups(optim, lrs)
    return optim
