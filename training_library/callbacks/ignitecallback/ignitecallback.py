from ..callback import Callback
from ignite.metrics import Accuracy, Recall, Precision

class IgniteCallback(Callback):
    order = 9

    def __init__(self):
        super(IgniteCallback, self).__init__()
        self.metrics_classes = [Accuracy(), Recall(), Precision()]
        self.metric = None
        self.base_name = 'train_'

    def begin_fit(self):
        for key in self.run.metrics.keys():
            for metric in self.metrics_classes:
                self.run.metrics[key][metric.__class__.__name__] = []

    def begin_epoch(self):
        self.reset()
        self.metric = self.run.metrics['train']

    def begin_eval(self):
        self.reset()
        self.metric = self.run.metrics['eval']

    def after_loss(self):
        for metric in self.metrics_classes:
            metric.update((self.run.y_hat, self.run.y))

    def after_all_batches(self):
        for metric in self.metrics_classes:
            self.metric[metric.__class__.__name__].append(metric.compute())

    def reset(self):
        for metric in self.metrics_classes:
            metric.reset()
