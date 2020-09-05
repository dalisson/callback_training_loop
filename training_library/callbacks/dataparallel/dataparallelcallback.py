from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.DistributedDataParallel as DistributedDataParallel 
from ..callback import Callback


class ParallelTrainer(Callback):
    #run_after,run_before = TrainEvalCallback,Recorder
    fup = None # for `find_unused_parameters` in DistributedDataParallel()
    def __init__(self, cuda_ids=0, sync_bn=True):
        super(ParallelTrainer, self).__init__()
        self.old_dls = None
        self.world_size = len(self.cuda_ids)
        
    def init_config(self):
        