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

class AverageGrad(Stat):
    '''
    the moving average of the grads, used for optimizers that use momentum
    '''
    def init_state(self, p):
        return {'grad_avg' : torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom=0.9, **kwargs):
        state['grad_avg'].mul_(mom).add_(p.grad.data)
        return state