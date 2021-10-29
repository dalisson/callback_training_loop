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
        self.records['momentum'] = [[] for _ in range(n_groups)]
        self.records['loss'] = []
        self.records['batch_loss'] = []
        self.records['train_loss'] = []
        self.records['test_loss'] = []

    def after_loss(self):
        '''
        Register parameters after calculating loss
        '''
        last_loss = self.run.loss.detach().cpu().numpy().item()
        self.records['batch_loss'].append(last_loss)
        # loss is not a moving mean with window of size 10
        self.records['loss'].append(mean(self.records['batch_loss'][-10:]))
        if self.in_train:
            for i, param_group in enumerate(self.optim.param_groups):
                self.records['lr'][i].append(param_group['lr'])
                self.records['train_loss'].append(last_loss)
                try:
                    self.records['momentum'][i].append(param_group['momentum'])
                except:
                    self.records['momentum'][i].append(0)
        else:
            self.records['test_loss'].append(last_loss)

    def plot_lr(self, save=False, param_group_id=-1):
        '''
        Plots the learning of given param_group
        '''
        title = "Lr pg: {}".format(param_group_id)
        fig = self._do_plot(self.records['lr'][param_group_id], title)
        if save:
            fig.savefig("lr.png")

    def plot_momentum(self, save=False, param_group_id=-1):
        '''
        Plots the momentumentum of a given param_group
        '''
        title = "Momentum pg: {}".format(param_group_id)
        fig = self._do_plot(self.records['momentum'][param_group_id], title)
        if save:
            fig.savefig("momentum.png")

    def plot_loss(self, save=False):
        '''
        plots the loss along iterations
        '''
        fig = self._do_plot(self.records['loss'], "Full loss")
        if save:
            fig.savefig("loss.png")

    def plot_train_loss(self, save=True):
        '''
        Plot the train loss
        '''
        fig = self._do_plot(self.records['train_loss'], "Train loss")
        if save:
            fig.savefig("tr_loss.png")

    def plot_test_loss(self, save=True):
        '''
        Plot the test loss
        '''
        fig = self._do_plot(self.records['test_loss'], "Test loss")
        if save:
            fig.savefig("test_loss.png")

    def plot_metrics(self, figsize, save=False):
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
        if save:
            axes.savefig("Metrics.png")

    def plot_lr_find(self, skip_last=5):
        '''
        Plots the result of lr_finder
        '''
        assert len(self.records['loss']) > 0, 'No Records to Plot'
        n = len(self.records['loss'])-skip_last
        plt.plot(self.records['lr'][-1][:n], self.records['loss'][:n])
        plt.xscale('log')

    def _do_plot(self, data, title):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(data)
        plt.title(title)
        return fig
