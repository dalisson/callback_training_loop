from ..callback import Callback

class SetTrainableModulesCallback(Callback):

    def init_config(self):
        if hasattr(self.run.loss_func, 'parameters'):
            self.run.trainable_modules.append(self.run.loss_func)
