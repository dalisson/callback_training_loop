from ..callback import Callback

class CudaCallback(Callback):
    '''
    A callback to train on GPU
    '''
    order = 0
    def __init__(self, device=0):
        super(CudaCallback, self).__init__()
        self.device = device

    def begin_fit(self):
        '''
        sends the model to gpu
        '''
        self.model.to(self.device)

    def begin_batch(self):
        '''
        Sends the batches to gpu
        '''
        self.x_batch.to(self.device)
        self.y_batch.to(self.device)
