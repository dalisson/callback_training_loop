import wandb
from ..callback import Callback


class WandbCallback(Callback):
    '''
    Callback to allow monitoring the model through Wandb
    '''
    order = 10
    def __init__(self, configs, wandb_project, wandb_name, entity="minds"):
        super(WandbCallback, self).__init__()
        self.configs = configs
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.entity = entity
        self.stage = 'train'

    def begin_fit(self):
        wandb.init(name=self.wandb_name, project=self.wandb_project, entity=self.entity)
        config = wandb.config 
        for k, v in self.configs.items():
            setattr(config, k, v)
        wandb.watch(self.run.model, log='all')

    def begin_epoch(self):
        '''
        set staget to train
        '''
        self.stage = 'train'

    def begin_eval(self):
        '''
        set stage to eval
        '''
        self.stage = 'eval'

    def after_all_batches(self):
        '''
        Logs to wandbd after all batches are completed
        '''
        log = dict()
        for key in self.run.metrics[self.stage].keys():
            log[('%s_%s' % (self.stage, key))] = self.run.metrics[self.stage][key][-1]
        wandb.log(log)
