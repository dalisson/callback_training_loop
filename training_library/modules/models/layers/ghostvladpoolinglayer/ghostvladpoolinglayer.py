import torch
import torch.nn as nn
import torch.nn.functional as F

class VladPoolingLayer(nn.Module):
    '''
    Vlad Pooling layer, if ghost_clusters > 0 it becomes the ghost vlad
    '''
    def __init__(self, n_clusters=8, g_clusters=2, d_dim=512):
        super(VladPoolingLayer, self).__init__()
        self.output_dimension = d_dim * n_clusters
        self.n_clusters = n_clusters
        self.center_assignment = nn.Conv2d(in_channels=d_dim,
                                           out_channels=n_clusters+g_clusters,
                                           kernel_size=(7, 1),
                                           stride=(1, 1),
                                           bias=True)
        self.centroids = nn.Parameter(torch.rand(d_dim, n_clusters+g_clusters))

        self.features = nn.Sequential(nn.Conv2d(d_dim, d_dim, (7, 1), stride=(1, 1), bias=True),
                                      nn.ReLU(inplace=True))

    def forward(self, *inputs):
        features = self.features(inputs[0])
        cluster_score = self.center_assignment(inputs[0])
        #soft assignment
        soft_assign = F.softmax(cluster_score, dim=1)
        #getting the residuals with broadcasting
        broadcast_centroids = self.centroids.unsqueeze(-1).unsqueeze(-1)
        b_features = features.unsqueeze(2)
        residual = b_features - broadcast_centroids
        weighted = soft_assign.unsqueeze(1) * residual
        #weighted taking the real clusters
        cluster_res = weighted.sum((-2, -1))[..., :self.n_clusters]
        #normalization
        cluster_l2 = F.normalize(cluster_res, p=2, dim=1)
        #tranpose bz x clusters x D
        cluster_l2 = torch.transpose(cluster_l2, 2, 1)
        return cluster_l2.reshape(-1, self.output_dimension)