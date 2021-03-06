import torch

__all__ = ['listfy', 'compose', 'pairwise_distance']

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

def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances

def set_grad(m, b):
    '''
    Sets requires grad = bfor layers except for batchnorm layers
        m: nn.module class
        b: boolean 
    '''
    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)
