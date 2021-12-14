'''
Callback to calculate the normalized mean information
metric used for embedding learning
'''
import torch
from ...callback import Callback
from .utils import calc_normalized_mutual_information, cluster_by_kmeans,\
                   assign_by_euclidian_at_k, calc_recall_at_k
from statistics import mean

class NMIRecallCallback(Callback):
    '''
    A callback to calculate NMI and Recall
    '''
    order = 9
    def __init__(self, recall_levels: list = [1, 2, 4, 8], limit_batch=100):
        super().__init__()
        self.recall_levels = recall_levels
        self.recall_names = ['recall@%s' %level for level in recall_levels]
        self.emb = []
        self.targets = []
        self.limit_batch = limit_batch
        self.acc = 0
        self.all_nmi = []
        self.all_recall = [[] for _ in self.recall_levels]

    def begin_fit(self):
        '''
        Set up metrics at the beginning of fit
        '''
        for stage in self.metrics.keys():
            self.metrics[stage]['nmi'] = []
            for name in self.recall_names:
                self.metrics[stage][name] = []
                if stage == 'train':
                    self.metrics[stage][name] = [0]
            
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
        if self.acc < self.limit_batch:
            self.targets.extend(self.y_batch.detach().cpu().numpy())
            self.emb.extend(self.y_hat.detach().cpu().numpy())
            self.acc += 1
        else:
            self._calculate()
            self.acc = 0

    def after_all_batches(self):
        '''
        After all batches the metric is appended to the runner metrics
        '''
        if self.acc > 0:
            self._calculate
        nmi = mean(self.all_nmi)
        self.metrics[self.stage]['nmi'].append(nmi)
        for i, name in enumerate(self.recall_names):
            r_at_k = mean(self.all_recall[i])
            self.run.metrics[self.stage][name].append(r_at_k)
        self.emb, self.targets = [], []
        self.all_nmi = []
        self.all_recall = [[] for _ in self.recall_levels]

    def _calculate(self):
        n_classes = len(set(self.targets))
        nmi = calc_normalized_mutual_information(self.targets,
                                                 cluster_by_kmeans(self.emb, n_classes))
        self.all_nmi.append(nmi)
        Y = assign_by_euclidian_at_k(self.emb, self.targets, max(self.recall_levels))
        Y = torch.from_numpy(Y)
        # calculate recall at different levels
        for i, level in enumerate(self.recall_levels):
            r_at_k = calc_recall_at_k(self.targets, Y, level)
            self.all_recall[i].append(r_at_k)
        self.emb, self.targets = [], []
