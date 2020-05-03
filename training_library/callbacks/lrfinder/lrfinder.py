from ..callback import Callback
from ..exceptions import CancelTrainException

class LR_Find(Callback):
    order = 1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        super(LR_Find, self).__init__()
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.optim.param_groups:
            pg['lr'] = lr

    def after_optim_step(self):
        if self.iter >= self.max_iter or self.loss > self.best_loss*10:
            self.run.training_canceled = True
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss

    def after_fit(self):
        '''
        after fit the iteractions go back to zero
        '''
        self.run.iter = 0
