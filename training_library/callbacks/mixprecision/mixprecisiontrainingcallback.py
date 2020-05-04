import apex.fp16_utils as fp16
import torch
from .utils import get_master, to_master_grads, to_model_params
from ..callback import Callback

class MixedPrecision(Callback):
    order = 99
    def __init__(self, loss_scale=512, flat_master=False):
        super().__init__()
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.loss_scale, self.flat_master = loss_scale, flat_master
        self.model_pgs, self.master_pgs = None, None

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = self.master_pgs #Put those param groups inside our runner.

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
        self.run.pred = self.run.pred.float() #Compute the loss in FP32
    def after_loss(self):
        '''
        The loss must be scaled to avoid underflow in fp16
        '''
        self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_backward(self):
        '''
        The loss must be copied to the master model (fp32 copy of model) and
        unscaled
        '''
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None:
                    param.grad.div_(self.loss_scale)

    def after_step(self):
        '''
        The optimizer is disconnected from the model so we must zero the gradients
        '''
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)
