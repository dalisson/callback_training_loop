'''
Callback to calculate the normalized mean information
metric used for embedding learning
'''
from ...callback import Callback
from .utils import calc_normalized_mutual_information, cluster_by_kmeans,\
                   assign_by_euclidian_at_k, calc_recall_at_k

class NMIRecallCallback(Callback):
    '''
    A callback to calculate NMI and Recall
    '''
    order = 9
    def __init__(self, recall_levels: list = [1, 2, 4, 8]):
        super().__init__()
        self.recall_levels = recall_levels
        self.recall_names = [('recall@%s' %level) for level in recall_levels]
        self.emb = []
        self.targets = []

    def begin_fit(self):
        '''
        Set up metrics at the beginning of fit
        '''
        for stage in self.metrics.keys():
            self.metrics[stage]['nmi'] = []
            for name in self.recall_names:
                self.metrics[stage][name] = []

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
        if not self.in_train:
            self.targets.extend(self.y_batch.detach().cpu().numpy())
            self.emb.extend(self.y_hat.detach().cpu().numpy())

    def after_all_batches(self):
        '''
        After all batches the metric is appended to the runner metrics
        '''
        if not self.in_train:

            nmi = calc_normalized_mutual_information(self.targets,
                                                     cluster_by_kmeans(self.emb, self.n_classes))
            self.metrics[self.stage]['nmi'].append(nmi)
            Y = assign_by_euclidian_at_k(self.emb, self.targets, max(self.recall_levels))
            Y = torch.from_numpy(Y)
            #calculate recall at different levels
            for level, name in zip(self.recall_levels, self.recall_names):
                r_at_k = calc_recall_at_k(self.targets, Y, level)
                self.run.metrics[self.stage][name].append(r_at_k)
        else:
            self.metrics[self.stage]['nmi'] = [0]
        self.emb, self.targets = [], []
