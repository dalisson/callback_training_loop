'''
Sets the base for optimizers
'''
import copy
from ..utils import compose, listfy

__all__ = ['Optimizer', 'StatefulOptimizer']

class Optimizer():
    '''
    The base of all optimizers, basically there is one optimizer
    the difference resides on the steppers of each particular optimizer,
    this class does not support optimizers with state
    '''
    def __init__(self, params, steppers, **defaults):
        self.params = listfy(params)        
        self.steppers = listfy(steppers)
        self.param_groups = [{'params': list(param), **defaults} for param in self.params]

    def grad_params(self):
        '''
        Returns all the parameters along with the associated hyper parameters
        '''
        return [(p, {key : value for key, value in item.items() if key != 'params'})
                for item in self.param_groups for p in item['params'] if p.grad is not None]

    def zero_grad(self):
        '''
        Zeros all the gradients
        '''
        for p, _ in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        '''
        Performs a step of the optimizer
        '''
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)

class StatefulOptimizer(Optimizer):
    '''
    Optimizer that holds the state, used for optimizers that need state
    eg. sgd with momentum
    '''
    def __init__(self, params, steppers, stats=None, **defaults):
        self.stats = listfy(stats)
        super().__init__(params, steppers, **defaults)
        self.state = {}

    def step(self):
        for p, hyper in self.grad_params():
            if p not in self.state:
                self.state[p] = {key: item  for stat in self.stats for key, item
                                 in stat.init_state(p).items()}
            state = self.state[p]
            for stat in self.stats:
                state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, dic: dict):
        assert isinstance(dic, dict)
        try:
            equality = list(k for k in dic.keys()) == list(key for key, _ in self.grad_params())
            assert equality or dic == {}
        except Exception as e:
            return e
        self.state = dic
