'''
Stats for optimizers that require the state of gradients
'''
import torch
class Stat():
    '''
    base for stats used on the stateful optmize
    '''
    def init_state(self, p):
        '''
        the initial state
        '''
        raise NotImplementedError
    def update(self, p, state, **kwargs):
        '''
        updates the stats
        '''
        raise NotImplementedError

class AverageGradStat(Stat):
    '''
    the moving average of the grads, used for optimizers that use momentum
    '''
    def __init__(self, dampening: bool = True):
        self.dampening = dampening
    def init_state(self, p):
        return {'grad_avg' : torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom=0.9, **kwargs):
        state['mom_damp'] = 1-mom if self.dampening else 1
        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
        return state

class AverageSqrGradStat(Stat):
    '''
    the moving squared average of the grads, used for optimizers that use momentum
    '''
    def __init__(self, dampening: bool = True):
        self.dampening = dampening
    def init_state(self, p):
        return {'sqr_avg' : torch.zeros_like(p.grad.data)}
    def update(self, p, state, sqr_mom=0.99, **kwargs):
        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1
        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
        return state

class StepCountStat(Stat):
    '''
    Counts the steps of the optimizer, useful for debiasing
    '''
    def init_state(self, p):
        return {'step' : 0}
    def update(self, p, state, **kwargs):
        state['step'] += 1
        return state
