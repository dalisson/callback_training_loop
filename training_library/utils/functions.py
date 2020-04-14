__all__ = ['listfy']

def listfy(obj_to_list):
    '''
    returns a list of itens
    '''
    res = []
    for element in obj_to_list:
        if not isinstance(element, list):
            res += [element]
        else:
            res += listfy(element)
    return res