from ..callback import Callback

__all__ = ['ParamScheduler']

class ParamScheduler(Callback):
    order=1
    def __init__(self, pname, sched_func): 
        super(ParamScheduler, self).__init__()
        self.pname, self.sched_func = pname, sched_func

    def begin_fit(self):
        if not isinstance(self.sched_func, list):
            self.sched_func = [self.sched_func] * self.run.n_param_groups
        assert self.run.n_param_groups == len(self.sched_func)

    def set_param(self):
        for pg, func in zip(self.optim.param_groups, self.sched_func):
            pg[self.pname] = func(self.iter/self.total_iter)

    def begin_batch(self):
        if self.in_train: self.set_param()
