import torch
from ..callback import Callback
from ..exceptions import CancelTrainException


class FailOnNanCallback(Callback):
    #run_after,run_before = TrainEvalCallback,Recorder
    # for `find_unused_parameters` in DistributedDataParallel()
    order = 0
    def after_loss(self):
        if torch.isnan(self.run.loss):
            self.run.reason = "LOSS IS NAN"
            self.run.iteration = self.run.iter
            raise CancelTrainException

