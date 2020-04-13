from ..callback import Callback


class SetTrainEvalCallback(Callback):
    '''
    Alternate the model between training and evaluation
    '''
    def begin_fit(self):
        '''
        sets the model to training
        '''
        self.run.model.train()

    def begin_eval(self):
        '''
        sets the model to evaluation
        '''
        self.run.model.eval()
