from ..callback import Callback
from ignite.metrics import Accuracy, Recall, Precision

class IgniteCallback(Callback):
    order = 9

    def __init__(self):
        super(IgniteCallback, self).__init__()
        self.metrics = [Accuracy(), Recall(), Precision()]
        self.train_metrics = dict()
        self.eval_metrics = dict()
        self.metric = None
        self.base_name = 'train_'

    def begin_epoch(self):
        self.reset()
        self.metric = self.train_metrics
        self.base_name = 'train_'

    def begin_eval(self):
        self.reset()
        self.metric = self.eval_metrics
        self.base_name = 'eval_'

    def after_loss(self):
        if not self.run.output:
            self.run.metrics = None
            return
        for metric in self.metrics:
            metric.update((self.run.y_hat, self.run.y))

    def after_all_batches(self):
        for metric in self.metrics:
            self.metric[(self.base_name + metric.__class__.__name__)] = metric.compute()
        self.run.metrics = self.metric

    def reset(self):
        for metric in self.metrics:
            metric.reset()
