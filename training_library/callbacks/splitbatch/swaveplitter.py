from ..callback import Callback


class SWaveSpliterCallback(Callback):
    order = 0

    def begin_batch(self):
        self.run.x_batch, self.run.y_batch, self.complement = self.run.current_batch

    def after_pred(self):
        self.run.y_batch = [self.run.y_batch, self.complement]
