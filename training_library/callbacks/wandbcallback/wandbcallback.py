import wandb
from ..callback import Callback


class WandbCallback(Callback):
    order = 10
    def __init__(self, configs, wandb_project, wandb_name, entity="minds"):
        super(WandbCallback, self).__init__()
        self.configs = configs
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.entity = entity
        self.config = None
        self.metrics = None

    def begin_fit(self):
        wandb.init(name=self.wandb_name, project=self.wandb_project, entity=self.entity)
        config = wandb.config 
        for k, v in self.configs.items():
            setattr(config, k, v)
        self.config = config
        wandb.watch(self.run.model, log='all')

    def after_all_batches(self):
        wandb.log(self.run.metrics)
