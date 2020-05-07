'''
Callback for mix precision training
'''

import apex.fp16_utils as fp16
import torch
from .utils import get_master, to_master_grads, to_model_params, grad_overflow
from ..callback import Callback

class MixedPrecisionCallback(Callback):
    '''
    Callback allows training with mix precision floating points fp16 and fp32
    current version can use dynamic scaling for loss
    '''
    order = 99
    def __init__(self, loss_scale=512, flat_master=False, dynamic=True, max_loss_scale=2.**12,
                 div_factor=2., scale_wait=500):
        super().__init__()
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.loss_scale, self.flat_master = loss_scale, flat_master
        self.dynamic, self.max_loss_scale = dynamic, max_loss_scale
        self.div_factor, self.scale_wait = div_factor, scale_wait
        self.loss_scale = max_loss_scale if dynamic else loss_scale
        self.model_pgs, self.master_pgs = None, None
        self.count = None

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.optim, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.optim.param_groups = self.master_pgs #Put those param groups inside our runner.
        if self.dynamic:
            self.count = 0

    def after_fit(self):
        '''
        model goes back to fp32 after training
        '''
        self.model.float()

    def begin_batch(self):
        '''
        At the beginning of each batch the inputs go to fp16
        '''
        #Put the inputs to half precision
        self.run.x_batch = self.run.x_batch.half()

    def after_pred(self):
        '''
        The loss must be computed on fp32
        '''
        self.run.y_hat = self.run.y_hat.float() #Compute the loss in FP32
    def after_loss(self):
        '''
        The loss must be scaled to avoid underflow in fp16
        '''
        self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_loss_backward(self):
        '''
        The loss must be copied to the master model (fp32 copy of model) and
        unscaled
        '''
        if self.dynamic and grad_overflow(self.model_pgs):
            #Divide the loss scale by div_factor, zero the grad (after_step will be skipped)
            self.loss_scale /= self.div_factor
            self.model.zero_grad()
            return True

        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params['params']:
                if param.grad is not None:
                    param.grad.div_(self.loss_scale)

        #Check if it's been long enough without overflow
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_optim_step(self):
        '''
        The optimizer is disconnected from the model so we must zero the gradients
        '''
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)
