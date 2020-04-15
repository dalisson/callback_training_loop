from ..callback import Callback


__all__ = ['ParamScheduler']

class ParamScheduler(Callback):
    order=1
    def __init__(self, pname, sched_func): 
        super(ParamScheduler, self).__init__()
        self.pname,self.sched_func = pname,sched_func

    def set_param(self):
        for pg in self.opt.param_groups:
            pg[self.pname] = self.sched_func(self.iter/self.total_iter)

    def begin_batch(self): 
        if self.in_train: self.set_param()

