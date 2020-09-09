from torch.nn.parallel import DataParallel
from ..callback import Callback


class ParallelTrainerCallback(Callback):
    #run_after,run_before = TrainEvalCallback,Recorder
    # for `find_unused_parameters` in DistributedDataParallel()
    order = 0
    def __init__(self, device_ids):
        super(ParallelTrainerCallback, self).__init__()
        self.device_ids = device_ids

    def before_fit(self):
        self.run.model = DataParallel(self.run.model, device_ids=self.device_ids)

    def after_fit(self):
        self.run.model = self.run.model.module
