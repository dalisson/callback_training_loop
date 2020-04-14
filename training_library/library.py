from torch.optim import SGD, Adam
from .callbacks.cuda import CudaCallback
from .callbacks.lrfinder import LR_Find
from .callbacks.recorder import RecorderCallback
from .callbacks.modelsettings import SetTrainableModulesCallback,\
                                     SetOptimizerCallback, SetTrainEvalCallback
from .runner import Runner

__all__ = ['build_standard_runner']

STANDARD_CALLBACK_LIST = [CudaCallback(), RecorderCallback(), SetTrainEvalCallback(),
                          SetTrainableModulesCallback(), SetOptimizerCallback()]

def build_standard_runner(model, data, loss_func, optim='SGD', min_lr=1e-2, max_lr=None):
    if optim.lower() == 'sgd':
        optimizer = SGD
    elif optim.lower() == 'adam':
        optimizer = Adam

    if max_lr:
        pass

    return Runner(model, data, loss_func, optimizer, min_lr, cbs=STANDARD_CALLBACK_LIST)


def lr_find(runner):

    runner.fit(2, additional_cbs=[LR_Find()])

    return runner.recorder.records['lr']