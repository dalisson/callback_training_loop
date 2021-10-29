class CancelBatchException(Exception):
    '''
    Exception for control flow of batch
    '''
    def __init__(self):
        super().__init__('Batch canceled')

class CancelAllBatchesException(Exception):
    '''
    Exception for control flow of all batches
    '''
    def __init__(self):
        super().__init__('All Batches cancelled')

class CancelTrainException(Exception):
    '''
    Exception for control flow of training
    '''
    def __init__(self):
        super().__init__('Training Cancelled')
