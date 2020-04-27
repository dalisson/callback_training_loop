from functools import partial
from ..utils import CustomList, listfy

__all__ = ['Hook', 'Hooks', 'get_layers_stats', 'find_layers', 'layer_condition', 'do_telemetry']

class Hook():
    def __init__(self, f, layer):
        self.h = layer.register_forward_hook(partial(f, self))
    def __del__(self):
        self.remove()
    def remove(self):
        '''
        removes hook
        '''
        self.h.remove()

class Hooks(CustomList):
    '''
    Hook list
    '''
    def __init__(self, modules, f):
        super().__init__([Hook(f, l) for l in listfy(modules)])
    def __enter__(self, *args):
        return self
    def __exit__(self, *args):
        self.remove()
    def remove(self):
        '''
        Removes all hooks
        '''
        for hook in self:
            hook.remove()

def find_layers(model, cond, returned_layers):
    for layer in model.children():
        if cond(layer):
            returned_layers.append(layer)
        else:
            find_layers(layer, cond, returned_layers)

def layer_condition(layer, layers_of_interest):
    return isinstance(layer, layers_of_interest)

def do_telemetry(model, condition, function):
    modules = []
    find_layers(model, condition, modules)
    if modules:
        return Hooks(modules, function)
    return []

def get_layers_stats(model, layers_of_interest, func):
    condition = partial(layer_condition, layers_of_interest=layers_of_interest)
    hooks = do_telemetry(model, condition, func)
    return hooks
