'''
Builds the optimizers
'''
from .optimizer import StatefulOptimizer
from .stats import AverageGradStat, AverageSqrGradStat, StepCountStat
from .steppers import weight_decay_step, sgd_with_momentum_step, adam_step

__all__ = ['sgd', 'adam']

def sgd(model, lr, mom=0, weight_decay=1e-5):
    '''
    Builds a sgd optimizer
    '''
    return StatefulOptimizer(model.parameters(), [weight_decay_step, sgd_with_momentum_step],
                             [AverageSqrGradStat(dampening=False)], lr=lr, mom=mom, wd=weight_decay)

def adam(model, lr, beta=9e-1, weight_decay=1e-5):
    '''
    Builds the adam optimizer
    '''
    return StatefulOptimizer(model.parameters(), [weight_decay_step, adam_step],
                             [StepCountStat(), AverageGradStat(), AverageSqrGradStat()],
                             lr=lr, mom=beta, wd=weight_decay)
