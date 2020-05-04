from ..callback import Callback
from ..exceptions import CancelTrainException

class LR_Find(Callback):
    order = 2
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        super(LR_Find, self).__init__()
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9
        self.state_dict = []
        self.initial_lrs = self.run.lr

    def begin_fit(self):
        self.state_dicts.extend([self.model.state_dict(), self.optim.state_dict()])
        if hasattr(self.run.loss_func, 'parameters'):
            self.state_dicts.append(self.run.loss_func.state_dict())

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
         # o  state dict deve voltar ao original
        for item, s_dict in zip([self.model, self.optim, self.loss_func], self.state_dicts):
            item.load_state_dict(s_dict)
        self.run.lr = self.initial_lrs
        self.run.iter = 0
