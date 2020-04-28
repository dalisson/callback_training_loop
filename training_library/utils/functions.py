__all__ = ['listfy', 'compose']

def listfy(obj_to_list):
    '''
    returns a list of itens
    '''
    if not isinstance(obj_to_list, (tuple, list)):
        return [obj_to_list]
    res = []
    for element in obj_to_list:
        if not isinstance(element, list):
            res += [element]
        else:
            res += listfy(element)
    return res

def compose(x, funcs, **kwargs):
    '''
    Function composition
    '''
    assert isinstance(funcs, (list, tuple))
    for func in funcs:
        x = func(x, **kwargs)
    return x
