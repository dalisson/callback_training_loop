import logging

from torch.functional import meshgrid

from ..callback import Callback


class LoggerCallback(Callback):
    order = 20

    def __init__(self, filename):
        super(LoggerCallback, self).__init__()
        logging.basicConfig(level=logging.INFO, filename=filename)

    def begin_fit(self):
        msg = "Starting training, e %s" % self.run.epochs
        logging.info(msg)

    def begin_epoch(self):
        msg = "------------\n"
        msg += "Starting epoch, current epoch %s" % self.run.epoch
        logging.info(msg)

    def after_loss(self):
        msg = "iter {}\nloss {}".format(self.run.global_iter, self.run.loss)
        logging.info(msg)

    def after_batch(self):
        msg = "batch finished"
        logging.info(msg)

    def begin_validate(self):
        msg = "begging validation"
        logging.info(msg)

    def after_cancel_train(self):
        msg = "training canceled at epoch %s" % self.run.epoch
        if hasattr(self.run, 'reason'):
            msg += '\n {}'.format(self.run.reason)
        logging.critical(msg)

    def after_fit(self):
        msg = "training process finished at %s" % self.run.epoch
        logging.info(msg)
