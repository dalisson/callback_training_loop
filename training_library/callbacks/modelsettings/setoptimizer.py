from ..callback import Callback
from ..callbackutils import discriminative_lr_optimizer

class SetOptimizerCallback(Callback):
    '''
    sets the optimizer
    '''
    order = 2
    def __init__(self, **kwargs):
        super(SetOptimizerCallback, self).__init__()
        self.kwargs = kwargs

    def initial_config(self):
        '''
        sets up the optimizer to the model parameters,
        additionaly if the loss function has trainable parameters
        these a sent to optimizizer as well
        '''

        self.run.optim = discriminative_lr_optimizer(self.run.trainable_modules, self.run.lr,
                                                     self.run.optim, **self.kwargs)
        self.run.n_param_groups = len(self.run.optim.param_groups)
