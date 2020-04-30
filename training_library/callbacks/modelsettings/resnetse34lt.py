from ..callback import Callback

class Resnetse34(Callback):

    def begin_batch(self):
        self.run.x_batch = self.run.x_batch.permute((0,2,3,1)).squeeze(1)