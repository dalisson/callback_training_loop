__all__ = ['listfy']

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
