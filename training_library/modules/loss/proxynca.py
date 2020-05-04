'''
Implements the proxyNCA loss, used for embedding learning,
the main class uses label smoothing by default.
'''


import torch
import torch.nn.functional as F
from ...utils.functions import pairwise_distance

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = torch.FloatTensor(T).to(device=device)
    return T


class ProxyNCA(torch.nn.Module):
    '''
    Proxy_NCA Loss function
    Warning!!! This class contains trainable parameters
    '''
    def __init__(self, nb_classes, sz_embed, smoothing_const=0.1, softmax=F.log_softmax):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.smoothing_const = smoothing_const
        self.softmax = softmax

    def forward(self, X, T):

        P = self.proxies
        P = 3 * F.normalize(P, p=2, dim=-1)
        X = 3 * F.normalize(X, p=2, dim=-1)
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=self.smoothing_const
        )

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = torch.sum(- T * self.softmax(D, -1), -1)

        return loss.mean()
