'''
Callback for cancelling training after a few iteractions
useful for telemetry
'''
from ..exceptions import CancelTrainException
from ..callback import Callback

class CancelTrainCallback(Callback):
    '''
    Allows user to cancel training after a few iteractions
    '''
    def __init__(self, iteractions):
        super().__init__()
        self.n_iteractions = iteractions

    def after_batch(self):
        '''
        Cancel after batch
        '''
        if self.run.iter >= self.n_iteractions:
            raise CancelTrainException
