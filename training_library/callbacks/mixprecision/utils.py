'''
These are the functions used on the mixprecision training callback
Batch norm types are left in single precision follow nvidia recomendation

In torch:
    batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


flat_master option cant be used with parameters that have both parameters
that must be trained in fp32, eg. batch norm, and parameters trained in fp16
eg. nn.Linear, nn.Conv2d
'''
import apex.fp16_utils as fp16
import torch
from torch.nn.utils import parameters_to_vector

def grad_overflow(param_groups):
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                s = float(p.grad.data.float().sum())
                if s == float('inf') or s == float('-inf') or s != s: return True
    return False

def dictfy_pgs(p_groups, optim):
    '''
    Turns the parameters groups back into dictionaries
    '''
    return [{'params' : pg, **{key : value for key, value in hypers.items() if key != 'params'}}
            for pg, hypers in zip(p_groups, optim.param_groups)]

def get_master(opt, flat_master=False):
    '''
    Creates a copy of fp16 model parameters in fp32, it handles
    the optimizer with different parameter groups
    '''
    model_pgs = [[pm for pm in pg['params'] if pm.requires_grad] for pg in opt.param_groups]
    if flat_master:
        master_pgs = []
        for p_group in model_pgs:
            m_param = parameters_to_vector([param.data.float() for param in p_group])
            m_param = torch.nn.Parameter(m_param, requires_grad=True)
            if m_param.grad is None:
                m_param.grad = m_param.new(*m_param.size())
            master_pgs.append([m_param])
    else:
        master_pgs = [[param.clone().float().detach() for param in pg] for pg in model_pgs]
        for p_group in master_pgs:
            for param in p_group:
                param.requires_grad_(True)

    #parameter groups must be dictionaries for the optimizer
    model_pgs = dictfy_pgs(model_pgs, opt)
    master_pgs = dictfy_pgs(master_pgs, opt)

    return model_pgs, master_pgs

def to_master_grads(model_pgs, master_pgs, flat_master: bool = False)->None:
    '''
    Copys all the gradients to the fp32 (master parameters) of model, so that the
    optimizer sted can be performed in fp32
    '''
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        fp16.model_grads_to_master_grads(model_params['params'], master_params['params'],
                                         flat_master=flat_master)

def to_model_params(model_pgs, master_pgs, flat_master: bool = False)->None:
    '''
    After the optimizer step the fp32 (master parameters) of the model are copied back to
    the model in fp16
    '''
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        fp16.master_params_to_model_params(model_params['params'], master_params['params'],
                                           flat_master=flat_master)
