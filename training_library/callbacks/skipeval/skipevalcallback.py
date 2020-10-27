'''
Callback for skiptraining and going straigth to eval

'''

from ..callback import Callback
from ..exceptions import CancelAllBatchesException

class SkipEvalCallback(Callback):
    '''
    Skip the training step, useful when we only want to run
    the eval step
    '''
    def begin_all_batches(self):
        '''
        Cancel all batches for training and goes straight into eval
        '''
        if not self.in_train:
            raise CancelAllBatchesException

    def after_fit(self):
        '''
        iteractions go back to zero at the end of fit
        '''
        self.run.iter = 0
