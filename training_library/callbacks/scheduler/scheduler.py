'''
Scheduler for parameters such as the learning rate
'''

import torch
import math
from functools import partial

from ..callback import Callback

__all__ = ['one_cycle_scheduler', 'exp_scheduler']

class ParamScheduler(Callback):
    '''
    Scheduler for any parameter of optmizer
    '''
    order=1
    def __init__(self, pname, sched_func):
        super(ParamScheduler, self).__init__()
        self.pname, self.sched_func = pname, sched_func
        self.train_iter = 0

    def begin_fit(self):
        if not isinstance(self.sched_func, list):
            self.sched_func = [self.sched_func] * self.run.n_param_groups
        assert self.run.n_param_groups == len(self.sched_func)
        self.train_iter = len(self.data.train_dl)  * self.epochs

    def set_param(self):
        '''
        sets the value of the parameter
        '''
        for pg, func in zip(self.optim.param_groups, self.sched_func):
            pg[self.pname] = func(self.iter/self.train_iter)

    def begin_batch(self):
        '''
        Sets the parameter at each iteraction
        '''
        if self.in_train:
            self.set_param()
    @property
    def name(self):
        base_name = super().name
        return base_name+'_{}'.format(self.pname)

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner
@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)
@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner



def one_cycle_scheduler(lrs, n_param_groups, max_lr, min_mom=0.85, mom=0.95, sched_mom=True, divs=None):
    divs = [0.3, 0.7] if not divs else divs
    if not isinstance(max_lr, list):
        max_lr = [max_lr] * n_param_groups
    sched_funcs = []
    for base_lr, m_lr in zip(lrs, max_lr):
        func = combine_scheds(divs, [sched_cos(base_lr, m_lr), sched_cos(m_lr, base_lr*1e-1)])
        sched_funcs.append(func)

    lr_scheduler = ParamScheduler(pname='lr', sched_func=sched_funcs)

    #momentum scheduling
    if sched_mom:
        sched_funcs = list()
        base_moms = [mom]* n_param_groups
        for mom in base_moms:
            func = combine_scheds(divs, [sched_cos(mom, min_mom), sched_cos(min_mom, mom)])
            sched_funcs.append(func)

        mom_scheduler = ParamScheduler(pname='momentum', sched_func=sched_funcs)

    return lr_scheduler, mom_scheduler

def exp_scheduler(lrs, gamma, n_epochs):
    sched_funcs = []
    for lr in lrs:
        sched_funcs.append(sched_exp(lr, lr*(gamma**n_epochs)))
    ParamScheduler(pname='lr', sched_func=sched_funcs)
    return ParamScheduler
