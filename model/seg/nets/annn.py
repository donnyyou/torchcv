import torch.nn as nn
import torch.nn.functional as F

from model.backbones.backbone_selector import BackboneSelector
from model.tools.module_helper import ModuleHelper

from model.seg.utils.apnb import APNB
from model.seg.utils.afnb import AFNB


class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, configer):
        super(asymmetric_non_local_network, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
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

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x
