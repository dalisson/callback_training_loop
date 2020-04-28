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
        self.param_groups = list(params)
        #param_groups must be a list of lists
        if not isinstance(self.param_groups, list):
            self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]
        self.steppers = listfy(steppers)

    def grad_params(self):
        '''
        Returns all the parameters along with the associated hyper parameters
        '''
        return [(p, hyper) for pg, hyper in zip(self.param_groups, self.hypers)
                for p in pg if p.grad is not None]

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
