'''
Stepper functions to used on optimizer
'''

__all__ = ['weight_decay']

def maybe_update(os, dest, f):
    '''
    Replaces functions defaults if optimizer is provided
    '''
    for o in os:
        for k,v in f(o).items():
            if k not in dest:
                dest[k] = v

def weight_decay(p, lr, wd=0, **kwargs):
    '''
    weight decay stepper
    '''
    p.data.mul_(1- lr*wd)
    return p

