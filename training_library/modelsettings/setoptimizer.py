from ..callback import Callback
from ..utils import discriminative_lr_optimizer

class SetOptimizerCallback(Callback):
    '''
    sets the optimizer
    '''
    def __init__(self, **kwargs):
        super(SetOptimizerCallback, self).__init__()
        self.kwargs = kwargs

    def begin_fit(self):
        '''
        sets up the optimizer to the model parameters,
        additionaly if the loss function has trainable parameters
        these a sent to optimizizer as well
        '''
        trainable_modules = [self.run.model]
        if hasattr(self.run.loss_func, 'parameters'):
            trainable_modules.append(self.run.loss_func)
        self.run.optim = discriminative_lr_optimizer(trainable_modules, self.run.lr,
                                                     self.run.optim, **self.kwargs)
