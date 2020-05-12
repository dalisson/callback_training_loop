'''
Scheduler for parameters such as the learning rate
'''
from ..callback import Callback

__all__ = ['ParamScheduler']

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
