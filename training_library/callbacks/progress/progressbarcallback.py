from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from ..callback import Callback
from ..imports import partial


class ProgressbarCallback(Callback):
    order=0
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        self.iter_inside_epoch = 0

    def after_fit(self):
        self.mbar.on_iter_end()
    def after_batch(self): 
        self.pb.update(self.iter_inside_epoch)
        self.iter_inside_epoch += 1
        self.run.logger()
    def begin_epoch   (self): 
        self.iter_inside_epoch = 0
        self.set_pb()

    def begin_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.run.dl, parent=self.mbar)
        self.mbar.update(self.epoch)
