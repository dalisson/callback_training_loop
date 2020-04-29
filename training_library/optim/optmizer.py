'''
Sets the base for optimizers
'''
from ..utils import compose, listfy

class Optmizer():
    '''
    The base of all optimizers, basically there is one optimizer
    the difference resides on the steppers of each particular optimizer,
    this class does not support optimizers with state
    '''
    def __init__(self, params, steppers, **defaults):
        self.params = list(params)
        #param_groups must be a list of lists
        if not isinstance(self.params, list):
            self.params = [self.params]
        self.steppers = listfy(steppers)
        self.param_groups = [{'params': param, **defaults} for param in self.params]

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

class StatefulOptmizer(Optmizer):
    '''
    Optimizer that holds the state, used for optimizers that need state
    eg. sgd with momentum
    '''

