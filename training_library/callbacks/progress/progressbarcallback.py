from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from ..callback import Callback
from ..imports import partial


class ProgressbarCallback(Callback):
    order=20
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        self.iter_in_dl = 0

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.iter_in_dl += 1
        self.pb.update(self.iter_in_dl)

    def begin_epoch(self): 
        self.iter_in_dl = 0
        self.set_pb()
        self.pb.update(0)

    def begin_validate(self):
        self.iter_in_dl = 0
        self.set_pb()

    def after_all_batches(self):
        stage = 'train' if self.run.in_train else 'eval'
        stats = stage + ': '
        for k in self.run.metrics[stage].keys():
            stats += (k + '-' + self.run.metrics[stage][k][-1])
            stats += '|'
        self.mbar.write(stats[:-1])

    def set_pb(self):
        self.pb = progress_bar(self.run.dl, parent=self.mbar)
        self.mbar.update(self.epoch)
