'''
Stepper functions to used on optimizer
'''

__all__ = ['weight_decay_step', 'sgd_step', 'sgd_with_momentum_step']

def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p

def weight_decay_step(p, lr, wd=0, **kwargs):
    '''
    weight decay stepper
    '''
    p.data.mul_(1- lr*wd)
    return p

def sgd_with_momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)