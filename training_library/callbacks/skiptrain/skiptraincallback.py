'''
Callback for skiptraining and going straigth to eval

'''

from ..callback import Callback
from ..exceptions import CancelAllBatchesException

class SkipTrainCallback(Callback):
    '''
    Skip the training step, useful when we only want to run
    the eval step
    '''
    def begin_all_batches(self):

        if self.in_train:
            raise CancelAllBatchesException
