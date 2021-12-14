import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from statistics import mean

from ...callback import Callback


class EERCallback(Callback):
    order = 9

    def __init__(self, n_batches=100):
        super(EERCallback, self).__init__()
        self.embeddings = []
        self.labels = []
        self.n_batches = n_batches
        self.acc = 0
        self.all_eers = []

    def begin_fit(self):
        for stage in self.run.metrics.keys():
            self.run.metrics[stage]["eer"] = []

    def after_pred(self):
        if self.acc < self.n_batches:
            self.embeddings.append(self.run.y_hat.detach().cpu())
            self.labels.extend(self.run.y_batch.detach().cpu().numpy())
            self.acc += 1
        else:
            self._cal_eer()

    def after_all_batches(self):
        if self.acc > 0:
            self._cal_eer()
        eer = mean(self.all_eers)
        self.metrics[self.stage]["eer"].append(eer)

    def _cal_eer(self):
        sims = self._calc_cosine_sim()
        # Broadcast programming
        self.labels = torch.from_numpy(np.array(self.labels))
        equal_labels_index = self.labels[None] == self.labels[..., None]
        different_labels_index = (equal_labels_index == False)
        equal_labels_index = self._apply_mask(equal_labels_index)
        different_labels_index = self._apply_mask(different_labels_index)
        # similarities for same label
        equal_sims = sims[equal_labels_index].detach().numpy()
        # similarities for different label
        different_sims = sims[different_labels_index].detach().numpy()
        eer = self._eer(equal_sims, different_sims)
        self.all_eers.append(eer)
        self.embeddings, self.labels = [], []

    def _calc_cosine_sim(self):
        '''
        Calc pairwise similarities
        '''
        matrix_of_vectors = torch.cat(self.embeddings)
        dot_product = matrix_of_vectors@matrix_of_vectors.t()
        norms = torch.sqrt(torch.einsum('ii->i', dot_product))
        similarities = dot_product/(norms[None]*norms[..., None])

        return similarities

    def _apply_mask(self, indexes):
        '''
        Remove duplicate similarities
        '''
        ind = np.diag_indices(indexes.shape[0])
        indexes[ind[0], ind[-1]] = False

        return torch.triu(indexes, diagonal=1)

    def _eer(self, match_sims, diff_sims):
        '''
        eer calculation
        match_sims : np vector matching similarities
        diff_sims : np vector different similarities
        '''
        y_true = [1] * len(match_sims) + [0] * len(diff_sims)
        preds = np.append(match_sims, diff_sims)
        fpr, tpr, thresholds = roc_curve(y_true, preds)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
