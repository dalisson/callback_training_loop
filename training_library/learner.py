from torch.optim import SGD, Adam
from .callbacks.cuda import CudaCallback
from .callbacks.lrfinder import LR_Find
from .callbacks.recorder import RecorderCallback
from .callbacks.modelsettings import SetTrainableModulesCallback,\
                                     SetOptimizerCallback, SetTrainEvalCallback

from .callbacks.scheduler import ParamScheduler
from .callbacks.scheduler import sched_lin, sched_cos, sched_exp, combine_scheds
from .callbacks.progress import ProgressbarCallback
from .callbacks.splitloss import SplitLossCallback
from .callbacks.wandbcallback import WandbCallback
from .callbacks.ignitecallback import IgniteCallback
from .callbacks.savemodel import SaveOnEpochEndCallback
from .runner import Runner


__all__ = ['Learner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), SetTrainEvalCallback(),
                          SetTrainableModulesCallback(), ProgressbarCallback(),
                          SplitLossCallback(), IgniteCallback()]

class Learner(Runner):

    @property
    def learning_rate(self):
        lr = []
        for pg in self.optim.param_groups:
            lr.append(pg['lr'])
        return lr

    @learning_rate.setter
    def learning_rate(self, new_lr):
        if not isinstance(new_lr, (list, tuple)):
            new_lr = [new_lr] * self.n_param_groups
        for lr, pg in zip(new_lr, self.optim.param_groups):
            pg['lr'] = lr
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

    @classmethod
    def build_standard_runner(cls, model, data, loss_func, optim='SGD', min_lr=1e-2, max_lr=None):
        '''
            Build a runner using standard callbacks
        '''
        if optim.lower() == 'sgd':
            optimizer = SGD
            STANDARD_CALLBACK_LIST.append(SetOptimizerCallback(momentum=9e-1, weight_decay=5e-4))
        elif optim.lower() == 'adam':
            optimizer = Adam
            STANDARD_CALLBACK_LIST.append(SetOptimizerCallback())
        if max_lr:
            pass

        return cls(model, data, loss_func, optimizer, min_lr, cbs=STANDARD_CALLBACK_LIST)

    def lr_find(self, skip_last=5):
        '''
        Finds the best learning rate for model

        '''
        state_dicts = []
        state_dicts.extend([self.model.state_dict(), self.optim.state_dict()])
        if hasattr(self.loss_func, 'parameters'):
            state_dicts.append(self.loss_func.state_dict())

        self.fit(1, additional_cbs=[LR_Find()])
        # o  state dict deve voltar ao original
        for component, s_dict in zip([self.model, self.optim, self.loss_func], state_dicts):
            component.load_state_dict(s_dict)

        self.remove_callback('lr_find')
        attr = getattr(self, 'recorder', None)
        if not attr:
            return 'recorder not found'
        attr.plot_lr_find(skip_last=skip_last)


    def fit_one_cycle(self, n_epochs, max_lr, divs=[0.3, 0.7], sched_type='cosine'):
        '''
        One cycle fitting using cosine scheduling.
        '''
        assert sum(divs) == 1
        if sched_type == 'cosine':
            sched_func = sched_cos
        elif sched_type == 'linear':
            sched_func = sched_lin
        else:
            print('undefined schedule function')
            return
        lrs = [group['lr'] for group in self.optim.param_groups]
        if not isinstance(max_lr, list):
            max_lr = [max_lr] * self.n_param_groups
        sched_funcs = []
        for base_lr, m_lr in zip(lrs, max_lr):
            func = combine_scheds(divs, [sched_func(base_lr, m_lr), sched_func(m_lr, base_lr*1e-1)])
            sched_funcs.append(func)

        sched_callback = ParamScheduler(pname='lr', sched_func=sched_funcs)
        self.remove_callback('paramscheduler')
        super().fit(epochs=n_epochs, additional_cbs=sched_callback)


    def fit_exp(self, n_epochs, gamma = 0.9):

        lrs = self.learning_rate
        sched_funcs = []
        for lr in lrs:
            sched_funcs.append(sched_exp(lr, lr*(gamma**n_epochs)))
        self.remove_callback('paramscheduler')
        super().fit(epochs=n_epochs,
                    additional_cbs=ParamScheduler(pname='lr', sched_func=sched_funcs))

    def add_wandb(self, configs, project, name, entity='minds'):
        '''
        Add callback to monitor project on wandb
        '''
        wandbc_b = WandbCallback(configs=configs,
                                 wandb_project=project,
                                 wandb_name=name,
                                 entity=entity)
        self.add_callbacks([wandbc_b])

    def save_every_epoch(self, optimizer=False):
        self.add_callbacks(SaveOnEpochEndCallback(optimizer=optimizer))
