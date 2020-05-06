'''
Mother of all callbacks this is the basic callback class
'''

import re

__all__ = ['Callback']

class Callback(object):
    '''
    Base callback class
    '''
    order = 0
    def __init__(self):
        self.run = None
    def set_runner(self, run):
        '''
        Sets the object runnig the callback
        '''
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        '''
        sets the name of the callback
        '''
        name = re.sub(r'Callback$', '', self.__class__.__name__).lower()
        return name
    
    def begin_fit(self):
        '''
        beginning fit process
        '''

    def begin_epoch(self):
        '''
        beginning an epoch process
        '''
    def __repr__(self):
        return self.run.model.__class__.__name__ + '.' + self.name

    def __call__(self, cb_name):
        attribute = getattr(self, cb_name, None)
        return attribute()
