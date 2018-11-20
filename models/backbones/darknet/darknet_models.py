import os
import torch
import torch.nn as nn
import math
from collections import OrderedDict

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from utils.tools.logger import Logger as Log


model_urls = {
    'darknet21': 'https://download.pytorch.org/models/darknet53_weights_pytorch.pth',
    'darknet53': 'https://download.pytorch.org/models/darknet53_weights_pytorch.pth',
}


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class DarkNetModels(object):

    def __init__(self, configer):
        self.configer = configer

    def darknet21(self):
        """Constructs a darknet-21 model.
        """
        model = DarkNet([1, 1, 2, 2, 1])
        if self.configer.get('network', 'pretrained') or self.configer.get('network', 'pretrained_model') is not None:
            if self.configer.get('network', 'pretrained_model') is not None:
                Log.info('Loading pretrained model:{}'.format(self.configer.get('network', 'pretrained_model')))
                pretrained_dict = torch.load(self.configer.get('network', 'pretrained_model'))
            else:
                pretrained_dict = self.load_url(model_urls['darknet21'])

            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys:{}'.format(matched_dict.keys()))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)

        return model


    def darknet53(self):
        """Constructs a darknet-53 model.
        """
        model = DarkNet([1, 2, 8, 8, 4])
        if self.configer.get('network', 'pretrained') or self.configer.get('network', 'pretrained_model') is not None:
            if self.configer.get('network', 'pretrained_model') is not None:
                Log.info('Loading pretrained model:{}'.format(self.configer.get('network', 'pretrained_model')))
                pretrained_dict = torch.load(self.configer.get('network', 'pretrained_model'))
            else:
                pretrained_dict = self.load_url(model_urls['darknet53'])

            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys:{}'.format(matched_dict.keys()))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)

        return model

    def load_url(self, url, map_location=None):
        model_dir = os.path.join(self.configer.get('project_dir'), 'models/backbones/darknet/pretrained')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))

        return torch.load(cached_file, map_location=map_location)