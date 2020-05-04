from .callbacks.cuda import CudaCallback
from .callbacks.exceptions import DeviceException
from .callbacks.ignitecallback import IgniteCallback
from .callbacks.lrfinder import LR_Find
from .callbacks.recorder import RecorderCallback
from .callbacks.mixprecision import MixedPrecisionCallback
from .callbacks.scheduler import ParamScheduler
from .callbacks.scheduler import sched_lin, sched_cos, sched_exp, combine_scheds
from .callbacks.splitloss import SplitLossCallback
from .callbacks.progress import ProgressbarCallback
from .callbacks.savemodel import SaveOnEpochEndCallback
from .callbacks.wandbcallback import WandbCallback

from .runner import Runner


__all__ = ['Learner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), ProgressbarCallback(),
                          SplitLossCallback(), IgniteCallback()]

class Learner(Runner):
    '''
    The learner class
    '''
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
    def build_standard_learner(cls, model, data, loss_func, optim):
        '''
        Build a runner using standard callbacks
        '''
        return cls(model, data, loss_func, optim, cbs=STANDARD_CALLBACK_LIST)

    def lr_find(self, skip_last=5):
        '''
        Finds the best learning rate for model
        '''
        self.fit(1, additional_cbs=[LR_Find()])
        attr = getattr(self, 'recorder', None)
        if not attr:
            return 'recorder not found'
        attr.plot_lr_find(skip_last=skip_last)
        self.remove_callback('lr_find')

    def fit_one_cycle(self, n_epochs, max_lr, divs=None, sched_type='cosine'):
        '''
        One cycle fitting using cosine scheduling.
        '''
        divs = [0.3, 0.7] if not divs else divs
        assert sum(divs) == 1
        if sched_type == 'cosine':
            sched_func = sched_cos
        elif sched_type == 'linear':
            sched_func = sched_lin
        else:
            print('undefined schedule function')
            return
        lrs = self.lr
        if not isinstance(max_lr, list):
            max_lr = [max_lr] * self.n_param_groups
        assert len(max_lr) == self.n_param_groups
        sched_funcs = []
        for base_lr, m_lr in zip(lrs, max_lr):
            func = combine_scheds(divs, [sched_func(base_lr, m_lr), sched_func(m_lr, base_lr*1e-1)])
            sched_funcs.append(func)

        sched_callback = ParamScheduler(pname='lr', sched_func=sched_funcs)
        self.remove_callback('paramscheduler')
        super().fit(epochs=n_epochs, additional_cbs=sched_callback)


    def fit_exp(self, n_epochs, gamma=0.9):
        '''
        Fits on exponencial learning rate
        '''
        lrs = self.lr
        sched_funcs = []
        for lr in lrs:
            sched_funcs.append(sched_exp(lr, lr*(gamma**n_epochs)))
        self.remove_callback('paramscheduler')
        super().fit(epochs=n_epochs,
                    additional_cbs=ParamScheduler(pname='lr', sched_func=sched_funcs))

    def wandb_logger(self, configs, project, name, entity='minds'):
        '''
        Add callback to monitor project on wandb
        '''
        wandbc_b = WandbCallback(configs=configs,
                                 wandb_project=project,
                                 wandb_name=name,
                                 entity=entity)
        self.remove_callback('wandb')
        self.add_callbacks([wandbc_b])

    def save_every_epoch(self, optimizer=False):
        '''
        Backup the model at each epoch
        '''
        self.add_callbacks(SaveOnEpochEndCallback(optimizer=optimizer))

    def to_half(self, loss_scale=512, flat_master=False):
        '''
        Set the training to mix precision floating points
        '''
        self.add_callbacks(MixedPrecisionCallback(loss_scale, flat_master))
