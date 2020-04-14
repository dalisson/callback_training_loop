from ..callback import Callback
from ..imports import torch
class CudaCallback(Callback):
    '''
    A callback to train on GPU
    '''
    order = 1
    def __init__(self, device=0):
        super(CudaCallback, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'

    def begin_fit(self):
        '''
        sends the model to gpu
        '''
        for module in self.run.trainable_modules:
            module.to(self.device)

    def begin_batch(self):
        '''
        Sends the batches to gpu
        '''
        self.run.x_batch = self.run.x_batch.to(self.device)
        self.run.y_batch = self.run.y_batch.to(self.device)
