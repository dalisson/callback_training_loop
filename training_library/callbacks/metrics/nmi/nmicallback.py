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
    def begin_fit(self):
        for stage in self.metrics.keys():
            self.metrics[stage]['nmi'] = []

    def begin_all_batches(self):
        '''
        at the beginning of all batches
        '''
        self.n_classes = len(self.dl.dataset.classes)
        self.emb = []
        self.targets = []

    def after_batch(self):
        '''
        compute the nmi at every batch
        '''
        self.targets.extend(self.y_batch.detach().cpu().numpy())
        self.emb.extend(self.y_hat.detach().cpu().numpy())

    def after_all_batches(self):
        '''
        After all batches the metric is appended to the runner metrics
        '''

        nmi = calc_normalized_mutual_information(self.targets,
                                                 cluster_by_kmeans(self.emb, self.n_classes))

        self.metrics[self.stage]['nmi'].append(nmi)
