from ..callback import Callback


class SetTrainEvalCallback(Callback):
    '''
    Alternate the model between training and evaluation
    '''
    order = 0
    def begin_epoch(self):
        '''
        sets the model to training
        '''
        self.run.model.train()

    def begin_validate(self):
        '''
        sets the model to evaluation
        '''
        self.run.model.eval()
