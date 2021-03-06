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

    def begin_fit(self):
        wandb.init(name=self.wandb_name, project=self.wandb_project, entity=self.entity)
        config = wandb.config 
        for k, v in self.configs.items():
            setattr(config, k, v)
        wandb.watch(self.run.model, log='all')

    def after_epoch(self):
        '''
        Logs to wandbd after each epoch is done
        '''
        log = dict()
        for stage in self.run.metrics.keys():
            for key in self.run.metrics[stage].keys():
                if self.run.metrics[stage][key]:
                    log[('%s_%s' % (stage, key))] = self.run.metrics[stage][key][-1]
        wandb.log(log)

    def after_fit(self):
        '''
        wandb wont accept a model running twice
        '''
        self.run.remove_callback('wandb')
