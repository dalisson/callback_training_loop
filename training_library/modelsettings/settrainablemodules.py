from ..callback import Callback

class SetTrainableModulesCallback(Callback):

    def begin_fit(self):
        if hasattr(self.run.loss_func, 'parameters'):
            self.run.trainable_modules = self.run.trainable_modules.append(self.run.loss_func)
