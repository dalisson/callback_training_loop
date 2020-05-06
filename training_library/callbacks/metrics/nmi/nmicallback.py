'''
Callback to calculate the normalized mean information
metric used for embedding learning
'''
from ...callback import Callback
from .utils import calc_normalized_mutual_information, cluster_by_kmeans



class NMICallback(Callback):
    '''
    A callback to calculate NMI
    '''
    order = 100
    def __init__(self, embedding_size=512):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_classes = 0
        self.emb = None
        self.targets = None

    def begin_fit(self):
        for stage in self.metrics.keys():
            self.metrics[stage]['nmi'] = []

    def begin_all_batches(self):
        '''
        at the beginning of all batches
        '''
        self.n_classes = self.data.n_classes
        self.emb = [[] * self.embedding_size] * len(self.dl.dataset)
        self.targets = [] * len(self.dl.dataset)

    def after_batch(self):
        '''
        compute the nmi at every batch
        '''
        self.targets.extend(self.y_batch.detach().cpu())
        self.emb.extend(self.y_hat.detach().cpu())

    def after_all_batches(self):
        '''
        After all batches the metric is appended to the runner metrics
        '''

        nmi = calc_normalized_mutual_information(self.targets,
                                                 cluster_by_kmeans(self.emb, self.n_classes))

        self.metrics[self.stage]['nmi'].append(nmi)
