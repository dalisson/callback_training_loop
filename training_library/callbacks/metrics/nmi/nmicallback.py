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
        self.n_classes = self.data.n_classes
        self.temp_holder = []
        for stage in self.metrics.keys():
            self.metrics[stage]['nmi'] = []

    def after_batch(self):
        '''
        compute the nmi at every batch
        '''
        targets = self.y_batch.cpu()
        embeddings = self.x_batch.cpu()
        nmi = calc_normalized_mutual_information(targets,
                                                 cluster_by_kmeans(embeddings, self.n_classes))
        self.temp_holder.append(nmi)

    def after_all_batches(self):
        '''
        After all batches the metric is appended to the runner metrics
        '''
        self.metrics[self.stage]['nmi'].append(self.temp_holder)
