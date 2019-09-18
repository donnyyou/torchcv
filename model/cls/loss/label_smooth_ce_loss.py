import torch
import torch.nn as nn


class LabelSmoothCELoss(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothCELoss, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / label.numel()
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        else:
            exit()

        return loss
