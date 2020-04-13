from ..callback import Callback
from ..imports import plt


class RecorderCallback(Callback):
    '''
    Records values along the iteractions
    '''

    def __init__(self):
        super(RecorderCallback, self).__init__()
        self.records = dict()

    def begin_fit(self):
        '''
        Sets everything up at the beginning of fit
        '''
        n_groups = len(self.run.optim.param_groups)
        self.records['lr'] = [[] for _ in range(n_groups)]
        self.records['loss'] = []

    def get_model(self):
        return self.model

    def after_loss(self):
        '''
        Register parameters after calculating loss
        '''
        self.records['loss'].append(self.run.loss)
        for i, param_group in enumerate(self.optim.param_groups):
            self.records['lr'][i].append(param_group['lr'])

    def plot_lr(self, param_group_id=-1):
        '''
        Plots the learning of given param_group
        '''
        plt.plot(self.record_parameters['lr'][param_group_id])

    def plot_loss(self):
        '''
        plots the loss along iterations
        '''
        plt.plot(self.recorded_parameters['loss'])
