'''
Base class for running the fully customizable training loop

'''
from ..callbacks.imports import torch
from ..callbacks.exceptions import CancelAllBatchesException,\
                                   CancelBatchException, CancelTrainException
from ..utils import listfy

__all__ = ['Runner']
__author__ = 'Dalisson Figueiredo'


class Runner():
    '''
    Handles a very customizable training loop through the use of callbacks

    '''

    def __init__(self, model, data, loss_func, optim, cbs=None):
        self.model, self.data, self.loss_func = model, data, loss_func
        self.optim = optim
        self.n_param_groups = len(optim.param_groups)
        self.training_canceled = False
        self.y_hat, self.x_batch, self.y_batch, self.loss = None, None, None, None
        self.epoch, self.epochs = 0, 0
        self.metrics = dict()
        self.metrics['train'], self.metrics['eval'] = dict(), dict()
        self.dl = None
        self.trainable_modules = [self.model]
        self.callbacks = []
        self.add_callbacks(cbs)
        self.iter = 0
        self.total_iter = 0
        self.in_train = True

        self('init_config')

    def add_callbacks(self, call_backs):
        '''
        adds callbacks to the runner callback list
        '''
        call_backs = listfy(call_backs)
        for c_b in call_backs:
            c_b.set_runner(self)
            setattr(self, c_b.name, c_b)
        self.callbacks += call_backs

    def remove_callback(self, cb_name):
        if hasattr(self, cb_name):
            delattr(self, cb_name)
            self.callbacks = [cb for cb in self.callbacks if cb.name != cb_name]


    def one_batch(self, x_b, y_b):
        '''
        Does batch of training loop

        '''
        try:
            self.x_batch, self.y_batch = x_b, y_b
            self('begin_batch')
            self.y_hat = self.model(self.x_batch)
            self('after_pred')
            self.loss = self.loss_func(self.y_hat, self.y_batch)
            self('after_loss')
            #stops execution when we are on validation
            if not self.in_train:
                return True
            self.loss.backward()
            self('after_loss_backward')
            self.optim.step()
            self('after_optim_step')
            self.optim.zero_grad()
            self.iter += 1
        except CancelBatchException:
            self('cancel_batch')
        finally:
            self('after_batch')

    def all_batches(self):
        '''
        Does all batches in training loop
        '''
        try:
            for x_batch, y_batch in self.dl:
                self.one_batch(x_batch, y_batch)
        except CancelAllBatchesException:
            self('cancel_all_batches')
        finally:
            self('after_all_batches')

    def begin_fit(self, epochs):
        '''
        begin fitting process
        '''
        self.epochs, self.loss = epochs, 0
        self.total_iter = len(self.data.train_dl) * epochs
        return self('begin_fit')

    def begin_epoch(self, current_epoch):
        '''
        begin epoch
        '''
        self.epoch = current_epoch
        self.dl = self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, additional_cbs=None):
        '''
        Does the fitting process
        '''
        if additional_cbs:
            self.add_callbacks(additional_cbs)
        try:
            self.begin_fit(epochs)
            for epoch in range(epochs):
                if self.begin_epoch(epoch):
                    self.in_train = True
                    self.model.train()
                    self.all_batches()
                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if self('begin_validate'):
                        self.in_train = False
                        self.model.eval()
                        self.all_batches()
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')


    def __call__(self, cb_name):
        for call_back in sorted(self.callbacks, key=lambda x: x.order):
            if hasattr(call_back, cb_name):
                res = call_back(cb_name)
                if not res and res is not None:
                    return False
        return True

    @property
    def lr(self):
        lr = []
        for pg in self.optim.param_groups:
            lr.append(pg['lr'])
        return lr

    @lr.setter
    def lr(self, new_lr):
        if not isinstance(new_lr, (list, tuple)):
            new_lr = [new_lr] * self.n_param_groups
        for lr, pg in zip(new_lr, self.optim.param_groups):
            pg['lr'] = lr

    def save(self, name=None, optimizer=False):
        '''
        Saves a model state dict and optionally the associated optimizer
            name: str - name of the model
            optimizer: bool - when true also save the optimizer state dict
        '''
        if name is None:
            name = 'model_e%s.' % self.epoch
            for metric in self.metrics['eval'].keys():
                name += '{}-{:.3f}.'.format(metric, self.metrics['eval'][metric][-1])
            name += 'pth'
        state_dict = dict()
        state_dict['model_state_dict'] = self.model.state_dict()
        if optimizer:
            state_dict['optimizer_state_dict'] = self.optim.state_dict()

        torch.save(state_dict, name)

    def load(self, model, optimizer=False):
        '''
        Loads parameters to model and optimizer
            name: Union[str, path] - model location
            optimizer: bool - is optimizer state to be loaded from file
        '''
        checkpoint = torch.load(model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
