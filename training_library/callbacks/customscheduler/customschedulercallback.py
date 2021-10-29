from ..callback import Callback


class CustomSchedulerCallback(Callback):
    '''
    Allows for the usage of other schedulers
    '''
    order = 1

    def __init__(self, scheduler, sched_on_iteration=False):
        super(CustomSchedulerCallback, self).__init__()
        self.scheduler = scheduler
        self.sched_on_iter = sched_on_iteration

    def after_batch(self):
        if self.run.in_train and self.sched_on_iter:
            self.scheduler.step()

    def after_epoch(self):
        if not self.sched_on_iter:
            self.scheduler.step()
