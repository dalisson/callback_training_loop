'''
Base class for running the fully customizable training loop

'''
from ..imports import *
from ..exceptions import * 

__all__ = ['Runner']
__author__ = 'Dalisson Figueiredo'




def listfy(obj_to_list):
    '''
    returns a list of itens
    '''
    res = []
    for element in obj_to_list:
        if not isinstance(element, list):
            res += [element]
        else:
            res += listfy(element)
    return res

class Runner():
    '''
    Handles a very customizable training loop through the use of callbacks

    '''

    def __init__(self, model, data, loss_func, optim, lr, cbs=None):
        self.model, self.data, self.loss_func = model, data, loss_func
        self.optim, self.lr = optim, lr
        self.y_hat, self.x_batch, self.y_batch, self.loss = None, None, None, None
        self.epoch, self.epochs = 0, 0
        self.call_backs = []
        self.add_callbacks(cbs)
        self.iter = 0
        self.in_train = True

    def add_callbacks(self, call_backs):
        '''
        adds callbacks to the runner callback list
        '''
        call_backs = listfy(call_backs)
        for c_b in call_backs:
            c_b.set_runner(self)
            setattr(self, c_b.name, c_b)
        self.call_backs += call_backs

    def one_batch(self, i, x_b, y_b):
        '''
        Does batch of training loop

        '''
        try:
            self.iter = i
            self.x_batch, self.y_batch = x_b, y_b
            self('begin_batch')
            self.y_hat = self.model(x_b)
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
        except CancelBatchException:
            self('cancel_batch')
        finally:
            self('after_batch')

    def all_batches(self, data):
        '''
        Does all batches in training loop
        '''
        try:
            for i, x_batch, y_batch in enumerate(data):
                self.one_batch(i, x_batch, y_batch)
                self('after_all_batches')
        except CancelAllBatchesException:
            self('cancel_all_batches')
        finally:
            self('after_all_batches')

    def begin_fit(self, epochs):
        '''
        begin fitting process
        '''
        self.epochs, self.loss = epochs, 0
        return self('begin_fit')

    def begin_epoch(self, current_epoch):
        '''
        begin epoch
        '''
        self.epoch = current_epoch
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
                    self.all_batches(self.data.train_dl)
                with torch.no_grad():
                    if self('begin_validate'):
                        self.in_train = False
                        self.all_batches(self.data.valid_dl)
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')


    def __call__(self, cb_name):
        for call_back in sorted(self.call_backs, key=lambda x: x.order):
            if hasattr(call_back, cb_name):
                print(call_back)
                res = call_back(cb_name)
                if not res and res is not None:
                    return False
        return True
