from torch.optim import SGD, Adam
from .callbacks.cuda import CudaCallback
from .callbacks.lrfinder import LR_Find
from .callbacks.recorder import RecorderCallback
from .callbacks.modelsettings import SetTrainableModulesCallback,\
                                     SetOptimizerCallback, SetTrainEvalCallback
from .callbacks.imports import plt, partial
from .callbacks.scheduler import ParamScheduler
from .callbacks.scheduler import sched_lin, sched_cos, sched_no, sched_exp, combine_scheds
from .callbacks.progress import ProgressbarCallback
from .runner import Runner


__all__ = ['Learner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), SetTrainEvalCallback(),
                          SetTrainableModulesCallback(), SetOptimizerCallback(),
                          ProgressbarCallback()]

class Learner(Runner):

    @property
    def device(self):
        '''
        The device running the learner
        '''
        attr = getattr(self, 'cuda')
        return attr.device

    @device.setter
    def device(self, new_device):
        attr = getattr(self, 'cuda')
        attr.device = new_device

    @classmethod
    def build_standard_runner(cls, model, data, loss_func, optim='SGD', min_lr=1e-2, max_lr=None):
        '''
            Build a runner using standard callbacks
        '''
        if optim.lower() == 'sgd':
            optimizer = SGD
        elif optim.lower() == 'adam':
            optimizer = Adam

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

        self.fit(2, additional_cbs=[LR_Find()])
        # o  state dict deve voltar ao original
        for component, s_dict in zip([self.model, self.optim, self.loss_func], state_dicts):
            component.load_state_dict(s_dict)

        self.remove_callback('lr_find')
        attr = getattr(self, 'recorder')
        if not attr:
            return 'recorder not found'
        lrs = attr.records['lr'][-1]
        loss = attr.records['loss']
        n = len(loss)-skip_last
        plt.plot(lrs[:n], loss[:n])
        plt.xscale('log')

    def fit_one_cycle(self, n_epochs, max_lr):

        lrs = [group['lr'] for group in self.optim.param_groups]

        sched_funcs = []
        for base_lr in lrs:
            func = combine_scheds([0.3, 0.7], [sched_cos(base_lr, max_lr), sched_cos(max_lr, base_lr*1e-1)])
            sched_funcs.append(func)

        sched_callback = ParamScheduler(pname='lr', sched_func=sched_funcs)
        self.remove_callback('paramscheduler')
        super().fit(epochs=n_epochs, additional_cbs=sched_callback)
