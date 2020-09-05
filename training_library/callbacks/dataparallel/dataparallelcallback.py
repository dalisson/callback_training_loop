import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import os
from ..callback import Callback


class ParallelTrainer(Callback):
    #run_after,run_before = TrainEvalCallback,Recorder
    # for `find_unused_parameters` in DistributedDataParallel()
    order = 0
    def __init__(self, n_devices=2):
        super(ParallelTrainer, self).__init__()
        self.old_dl = None
        self.world_size = n_devices

    def init_config(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9999'

    def begin_fit(self):
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=self.world_size,
                                rank=self.run.device)

        self.old_dl = (self.run.data.train_dl, self.run.data.valid_dl)
        self.run.data.train_dl = self._wrap_sampler(self.run.data.train_dl)
        self.run.data.valid_dl = self._wrap_sampler(self.run.data.valid_dl)
        self.run.model = DistributedDataParallel(self.run.model,
                                                 device_ids=[self.run.device])

    def _wrap_sampler(self, dl):
        dataset = dl.dataset
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.run.device)
        new_dataloader = DataLoader(dataset=self.data.train_dl.dataset,
                                    batch_size=self.run.data.bs,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=sampler)
        return new_dataloader

    def after_fit(self):
        self.run.model = self.run.model.module
