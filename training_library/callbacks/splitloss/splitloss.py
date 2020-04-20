from ..callback import Callback

class SplitLossCallback(Callback):
    '''
    Sometimes it is necessary to configure the loss to split both the loss and the logits
    this callback should set y_hat to be the logits used to compute the loss,
    it is necessay due to the way the am_softmax is configured

    '''
    order = 0
    def after_loss(self):
        if isinstance(self.run.loss, tuple):
            self.run.y_hat, self.run.loss = self.run.loss[-1], \
                                             self.run.loss[0]
        