from ..callback import Callback
from ..utils import discriminative_lr_optimizer

class SetOptimizerCallback(Callback):
    '''
    sets the optimizer
    '''
    def begin_fit(self):
        all_parameters = [self.run.model]
        if hasattr(self.run.loss_func, 'params'):
            all_parameters.append(self.run.loss_func)
        self.run.optim = discriminative_lr_optimizer(all_parameters, self.run.lr,
                                                     self.run.optim)
