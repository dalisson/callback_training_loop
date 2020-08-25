from ..callback import Callback

class SaveOnEpochEndCallback(Callback):

    def __init__(self, optimizer=False):
        super(SaveOnEpochEndCallback, self).__init__()
        self.optimizer = optimizer

    def after_epoch(self):
        self.run.save(optimizer=self.optimizer)
