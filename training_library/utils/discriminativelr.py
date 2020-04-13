from typing import Union
from ..imports import nn, torch

__all__ = ['discriminative_lr_optimizer']

def discriminative_lr_optimizer(network: Union[nn.Module, list], lr: Union[float, list], optimizer: torch.optim, **kwargs):
    '''
    Returns the optimizer with discriminative learning rates for each children on the network,
    if the number of children is different of the number of learning rates, the range
    between maximum and minimum learning learning rate will the distributed along
    the children, layers close to the base get smaller learning rates.

    input:
        netwok. torch.nn.Module
        lr: Union(float, list)
        optimizer: torch.optim.optimizer

        kwargs: weight_decay and others

     kwargs: weight_decay and others

    return
        torch.optim.optimizer
    '''
    if isinstance(network, list):
        model_children = [c for item in network for c in item.children()]
    else:
        model_children = list(network.children())
    n_groups = len(model_children)
    if isinstance(lr, list):
        #if there is a different number of learning rates and param groups in model
        if len(lr) != n_groups:
            min_lr, max_lr = min(lr), max(lr)
            step = (max_lr-min_lr)/(n_groups - 1)
            lrs = [(min_lr + x*step) for x in range(n_groups)]
        else:
            lrs = lr
    #if there is a single learning rate for the entire network
    else:
        lrs = [lr for _ in range(len(model_children))]
    #build parameters groups for optimization
    param_list = [dict(params=c.parameters(), lr=lr, **kwargs) for c, lr\
         in zip(model_children, lrs)]
    return optimizer(param_list)
