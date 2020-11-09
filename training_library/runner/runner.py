import importlib
from functools import partial

import torch.multiprocessing as mp

from ..callbacks.cuda import CudaCallback
from ..callbacks.dataparallel import ParallelTrainerCallback
from ..callbacks.exceptions import DeviceException
from ..callbacks.metrics import IgniteCallback
from ..callbacks.lrfinder import LR_Finder
from ..callbacks.recorder import RecorderCallback
from ..callbacks.scheduler import one_cycle_scheduler, exp_scheduler
from ..callbacks.progress import ProgressbarCallback
from ..callbacks.savemodel import SaveOnEpochEndCallback
from ..callbacks.skipeval import SkipEvalCallback
from ..callbacks.wandbcallback import WandbCallback
from ..callbacks.savemetricscallback import SaveMetricsCallback
from ..callbacks.failonnan import FailOnNanCallback

from .base import BaseRunner


ENABLE_HALF = False
if importlib.util.find_spec('apex'):
    ENABLE_HALF = True
    from ..callbacks.mixprecision import MixedPrecisionCallback


__all__ = ['Runner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), ProgressbarCallback(), FailOnNanCallback()]

class Runner(BaseRunner):
    '''
    The learner class
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit = super().fit

    @property
    def device(self):
        '''
        The device running the learner
        '''
        attr = getattr(self, 'cuda', None)
        if attr:
            return attr.device
        return 'cpu'

    @device.setter
    def device(self, new_device):
        attr = getattr(self, 'cuda', None)
        if attr:
            attr.device = new_device
        else:
            raise DeviceException

    @classmethod
    def create_standard_runner(cls, model, data, loss_func, optim):
        '''
        Build a runner using standard callbacks
        '''
        return cls(model=model,
                   data=data,
                   loss_func=loss_func,
                   optim=optim,
                   cbs=STANDARD_CALLBACK_LIST)

    def distribute_runner(self, device_ids):
        '''
        Distributes the learner among gpus
        '''
        self.remove_callback('paralleltrainer')
        self.device = device_ids[0]
        parallel_cb = ParallelTrainerCallback(device_ids=device_ids)
        self.add_callback(parallel_cb)

    def lr_find(self, skip_last=5):
        '''
        Finds the best learning rate for model
        '''
        self.fit(1, additional_cbs=[LR_Finder()])
        getattr(self, 'recorder', None).plot_lr_find(skip_last=skip_last)
        self.remove_callback('lr_finder')

    def fit_one_cycle(self, n_epochs, max_lr, supress_progress=False, **kwargs):
        '''
        One cycle fitting using cosine scheduling.
        '''

        lr_scheduler, mom_sched = one_cycle_scheduler(lrs=self.lr,
                                                      n_param_groups=self.n_param_groups,
                                                      max_lr=max_lr, **kwargs)
        self.remove_callback('paramscheduler_lr')
        self.remove_callback('paramscheduler_momentum')
        self.add_callback(lr_scheduler)
        self.add_callback(mom_sched)

        if supress_progress:
            self.remove_callback('progressbar')
        self.fit(epochs=n_epochs)


    def fit_exp(self, n_epochs, gamma=0.9, supress_progress=False):
        '''
        Fits on exponencial learning rate
        '''

        if supress_progress:
            self.remove_callback('progressbar')

        self.remove_callback('paramscheduler_lr')
        sched = exp_scheduler(lrs=self.lr, gamma=gamma, n_epochs=n_epochs)
        self.fit(epochs=n_epochs, additional_cbs=sched)

    def add_softmax_metrics(self):
        '''
        Adds the accuracy, recall and precision when training on softmax
        '''
        self.add_callback(IgniteCallback())

    def wandb_logger(self, configs, project, name, entity='minds'):
        '''
        Add callback to monitor project on wandb
        '''
        wandbc_b = WandbCallback(configs=configs,
                                 wandb_project=project,
                                 wandb_name=name,
                                 entity=entity)
        self.remove_callback('wandb')
        self.add_callback([wandbc_b])

    def save_every_epoch(self, optimizer=False):
        '''
        Backup the model at each epoch
        '''
        self.add_callback(SaveOnEpochEndCallback(optimizer=optimizer))

    def no_eval(self):
        '''
        Run the model through one epoch in the eval dataset
        '''
        self.remove_callback('skipeval')
        self.add_callback([SkipEvalCallback()])

    def half(self, loss_scale=512, dynamic=True, flat_master=False, **kwargs):
        '''
        Set the training to mix precision floating points
        '''
        if not ENABLE_HALF:
            return 'apex library not installed'
        self.add_callback(MixedPrecisionCallback(loss_scale, flat_master, dynamic, **kwargs))

    def add_csv_logger(self, f):
        self.remove_callback('savemetrics')
        cb = SaveMetricsCallback(f)
        self.add_callback(cb)
