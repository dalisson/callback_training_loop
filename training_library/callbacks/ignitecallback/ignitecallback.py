from ..callback import Callback
from ignite.metrics import Accuracy, Recall, Precision

class IgniteCallback(Callback):
    order = 9

    def __init__(self):
        super(IgniteCallback, self).__init__()
        self.metrics_classes = [Accuracy(), Recall(average=True), Precision(average=True)]

    def begin_fit(self):
        for key in self.run.metrics.keys():
            for metric in self.metrics_classes:
                self.run.metrics[key][metric.__class__.__name__] = []

    def after_loss(self):
        for metric in self.metrics_classes:
            metric.update((self.run.y_hat, self.run.y_batch))

    def after_all_batches(self):
        stage = 'train' if self.run.in_train else 'eval'
        for metric in self.metrics_classes:
            self.run.metrics[stage][metric.__class__.__name__].append(metric.compute())
        self.reset()

    def reset(self):
        for metric in self.metrics_classes:
            metric.reset()
