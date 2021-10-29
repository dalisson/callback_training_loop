from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from ..callback import Callback
from ..imports import partial


class ProgressbarCallback(Callback):
    order=20
    def begin_fit(self):
        self.epoch = 0
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        self.iter_in_dl = 0

    def begin_epoch(self):
        self.iter_in_dl = 0
        self.set_pb()
        self.pb.update(self.iter_in_dl)
        self.epoch += 1

    def after_loss(self):
        self.mbar.child.comment = 'Loss: {:.3f}'.format(self.recorder.records['loss'][-1])

    def after_batch(self):
        self.iter_in_dl += 1
        self.pb.update(self.iter_in_dl)

    def begin_validate(self):
        self.iter_in_dl = 0
        self.set_pb()
        self.pb.update(self.iter_in_dl)

    def after_all_batches(self):
        if self.run.training_canceled:
            self.mbar.write('Training cancelled at epoch %s - iter %s' % (self.epoch, self.run.iter))
            return True
        stats = 'Epoch {} - '.format(self.epoch)
        stats += self.stage + ': '
        for k in self.run.metrics[self.stage].keys():
            if self.metrics[self.stage][k]:
                stats += '{} - {:.2f} '.format(k, self.run.metrics[self.stage][k][-1])
                stats += ' '
        self.mbar.write(stats[:-1])

    def after_fit(self):
        self.mbar.on_iter_end()

    def set_pb(self):
        self.pb = progress_bar(self.run.dl, parent=self.mbar)
        self.mbar.update(self.epoch)
