'''
Stepper functions to used on optimizer
'''

__all__ = ['weight_decay_step', 'sgd_step', 'sgd_with_momentum_step', 'adam_step', 'lamb_step']

def sgd_step(p, lr, **kwargs):
    '''
    normal sgd step
    '''
    p.data.add_(-lr, p.grad.data)
    return p

def weight_decay_step(p, lr, wd=0, **kwargs):
    '''
    weight decay stepper
    '''
    p.data.mul_(1- lr*wd)
    return p


def sgd_with_momentum_step(p, lr, grad_avg, **kwargs):
    '''
    Sgd with momemtum step
    '''
    p.data.add_(grad_avg, alpha=-lr)
    return p


def debias(momentum, damp, step):
    '''
    helper function to compute debias
    '''
    return damp * (1-momentum**step)/(1-momentum)


def adam_step(p, lr, momentum, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps=1e-6, **kwargs):
    '''
    The adam step
    '''
    debias1 = debias(momentum, mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(grad_avg,
                    (sqr_avg/debias2).sqrt() + eps,
                    value=-lr/debias1)
    return p


def lamb_step(p, lr, momentum, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps=1e-6, wd=0, **kwargs):
    '''
    The lamb optimizer step, it has weight decay
    '''
    debias1 = debias(momentum,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    r1 = p.data.pow(2).mean().sqrt()
    step = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + wd*p.data
    r2 = step.pow(2).mean().sqrt()
    p.data.add_(-lr * min(r1/r2, 10), step)
    return p