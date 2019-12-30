import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.module_helper import ModuleHelper
from model.seg.utils.apnb import APNB
from model.seg.utils.afnb import AFNB
from model.seg.loss.loss import BASE_LOSS_DICT


class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, configer):
        super(asymmetric_non_local_network, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = ModuleHelper.get_backbone(
            backbone=self.configer.get('network.backbone'),
            pretrained=self.configer.get('network.pretrained')
        )
        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        self.fusion = AFNB(1024, 2048, 2048, 256, 256, dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
                         dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        x_ = data_dict['img']
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        x_dsn = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out_dict = dict(dsn_out=x_dsn, out=x)
        if self.configer.get('phase') == 'test':
            return out_dict

        loss_dict = dict()
        if 'dsn_ce_loss' in self.valid_loss_dict:
            loss_dict['dsn_ce_loss'] = dict(
                params=[x_dsn, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['dsn_ce_loss']])
            )

        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )

        if 'ohem_ce_loss' in self.valid_loss_dict:
            loss_dict['ohem_ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ohem_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ohem_ce_loss']])
            )
        return out_dict, loss_dict
