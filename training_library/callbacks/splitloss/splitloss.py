from ..callback import Callback

class SplitLossCallback(Callback):

    def after_loss(self):
        if isinstance(self.run.loss, tuple):
            self.run.output, self.run.loss = self.run.loss[-1], \
                                             self.run.loss[0]
        