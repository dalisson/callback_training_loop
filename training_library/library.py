from torch.optim import SGD, Adam
from .callbacks.cuda import CudaCallback
from .callbacks.lrfinder import LR_Find
from .callbacks.recorder import RecorderCallback
from .callbacks.modelsettings import SetTrainableModulesCallback,\
                                     SetOptimizerCallback, SetTrainEvalCallback
from .callbacks.imports import plt
from .runner import Runner


__all__ = ['Learner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), SetTrainEvalCallback(),
                          SetTrainableModulesCallback(), SetOptimizerCallback()]

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
        state_dict = self.model.state_dict()

        self.fit(2, additional_cbs=[LR_Find()])
        # o  state dict deve voltar ao original
        self.model.load_state_dict(state_dict)

        self.remove_callback('lr_find')
        attr = getattr(self, 'recorder')
        if not attr:
            return 'recorder not found'
        lrs = attr.records['lr'][-1]
        loss = attr.records['loss']
        n = len(loss)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], loss[:n])
        