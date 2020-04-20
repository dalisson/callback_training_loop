from ..callback import Callback
from ..imports import plt


class RecorderCallback(Callback):
    order = 10
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

    def after_loss(self):
        '''
        Register parameters after calculating loss
        '''
        self.records['loss'].append(self.run.loss.detach().cpu().numpy().item())
        for i, param_group in enumerate(self.optim.param_groups):
            self.records['lr'][i].append(param_group['lr'])

    def plot_lr(self, param_group_id=-1):
        '''
        Plots the learning of given param_group
        '''
        plt.plot(self.records['lr'][param_group_id])

    def plot_loss(self):
        '''
        plots the loss along iterations
        '''
        plt.plot(self.records['loss'])

    def plot_lr_find(self, skip_last=5):
        n = len(self.records['loss'])-skip_last
        plt.plot(self.records['lr'][-1][:n], self.records['loss'][:n])
        plt.xscale('log')
        self.begin_fit()
        