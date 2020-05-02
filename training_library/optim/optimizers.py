'''
Builds the optimizers
'''
from .optim import StatefulOptimizer
from .stats import AverageGradStat, AverageSqrGradStat, StepCountStat
from .steppers import weight_decay_step, sgd_with_momentum_step, adam_step,\
                      lamb_step

__all__ = ['sgd', 'adam', 'lamb']

def get_param_groups(net):
    groups = []
    for child in net.children():
        groups.append(list(child.parameters()))
    return groups

def sgd(model, lr, mom=0, weight_decay=1e-5):
    '''
    Builds a sgd optimizer
    '''

    return StatefulOptimizer(get_param_groups(model), [weight_decay_step, sgd_with_momentum_step],
                             [AverageSqrGradStat(dampening=False)], lr=lr, mom=mom, wd=weight_decay)

def adam(model, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5):
    '''
    Builds the adam optimizer
    '''
    assert isinstance(betas, tuple) and len(betas) == 2
    return StatefulOptimizer(get_param_groups(model), [weight_decay_step, adam_step],
                             [StepCountStat(), AverageGradStat(), AverageSqrGradStat()],
                             lr=lr, mom=betas[0], sqrt_mom=betas[1], wd=weight_decay)

def lamb(model, lr, betas: tuple = (9e-1, 99e-2), weight_decay=1e-5):
    '''
    Builds the adam optimizer
    '''
    assert isinstance(betas, tuple) and len(betas) == 2
    return StatefulOptimizer(get_param_groups(model), [weight_decay_step, lamb_step],
                             [StepCountStat(), AverageGradStat(), AverageSqrGradStat()],
                             lr=lr, mom=betas[0], sqrt_mom=betas[1], wd=weight_decay)
