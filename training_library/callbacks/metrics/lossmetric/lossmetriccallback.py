from ...callback import Callback
from statistics import StatisticsError, mean


class LossMetricCallback(Callback):
    order = 9

    def begin_fit(self):
        self.last_index = 0
        for key in self.run.metrics.keys():
            self.run.metrics[key]["avg_loss"] = []

    def after_all_batches(self):
        index = self.last_index - self.run.global_iter
        try:
            mean_loss = mean(self.run.recorder.records['batch_loss'][index:])
        except StatisticsError:
            mean_loss = 0
        self.run.metrics[self.stage]["avg_loss"].append(mean_loss)
        self.last_index = self.run.global_iter
