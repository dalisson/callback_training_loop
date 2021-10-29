from ..callback import Callback

class StandardSplitterCallback(Callback):
    order = 0
    def begin_batch(self):
        self.run.x_batch, self.run.y_batch = self.run.current_batch