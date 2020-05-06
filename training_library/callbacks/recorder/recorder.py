from ..callback import Callback
from ..imports import plt
from statistics import mean

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
        self.records['batch_loss'] = []

    def after_loss(self):
        '''
        Register parameters after calculating loss
        '''
        self.records['batch_loss'].append(self.run.loss.detach().cpu().numpy().item())
        #loss is not a moving mean with window of size 10
        self.records['loss'].append(mean(self.records['batch_loss'][-10:]))
        if self.in_train:
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

    def plot_metrics(self, figsize):
        '''
        Visualization for the model metrics
        '''
        n_rows = len(self.metrics['train'])
        figsize = (8, 8) if not figsize else figsize
        _, axes = plt.subplots(n_rows, 2, figsize=figsize)
        keys = [(s, m) for s in self.metrics.keys() for m in self.metrics[s].keys()]
        values = [self.metrics[stage][metric] for stage, metric in keys]
        for i, ax in enumerate(axes.flatten()):
            ax.plot(values[i])
            ax.set_title('%s_%s' % keys[i])

    def plot_lr_find(self, skip_last=5):
        '''
        Plots the result of lr_finder
        '''
        assert len(self.records['loss']) > 0, 'No Records to Plot'
        n = len(self.records['loss'])-skip_last
        plt.plot(self.records['lr'][-1][:n], self.records['loss'][:n])
        plt.xscale('log')
        