"""
@desc: the variety of center loss, which use the class weight as the class center and normalize both the weight and feature,
       in this way, the cos distance of weight and feature can be used as the supervised signal.
       It's similar with torch.nn.CosineEmbeddingLoss, x_1 means weight_i, x_2 means feature_i.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, scale):
        super(AgentCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        cos_dis = F.linear(F.normalize(x), F.normalize(self.centers)) * self.scale

        one_hot = torch.zeros_like(cos_dis)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # loss = 1 - cosine(i)
        loss = one_hot * self.scale - (one_hot * cos_dis)

        return loss.mean()
