import torch
import torch.nn as nn


# Defines the LightCNN generator.
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, input):
        x = self.filter(input)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input):
        x = self.conv_a(input)
        x = self.conv(x)
        return x

class LightCnnGenerator(nn.Module):
    def __init__(self, num_classes=99891, gpu_ids=[]):
        super(LightCnnGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
            x = F.dropout(feature, training=self.training)
            out = nn.parallel.data_parallel(self.fc2, x, self.gpu_ids)
        else:
            x = self.features(input)
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
            x = F.dropout(feature, training=self.training)
            out = self.fc2(x)
        return out, feature


class LightCnnFeatureGenerator(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(LightCnnFeatureGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)


    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
            # feature = F.dropout(feature, training=self.training)
        else:
            x = self.features(input)
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
            # feature = F.dropout(feature, training=self.training)
        return feature

class LightCnnNoFCFeatureGenerator(nn.Module):
    # output conv features(feature map of the last conv layer) of the light cnn model
    def __init__(self, gpu_ids=[]):
        super(LightCnnNoFCFeatureGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        # self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            feature = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
        else:
            feature = self.features(input)
        return feature

class LightCnnFC1Generator(nn.Module):
    # input is the feature map of conv layers
    def __init__(self, gpu_ids=[]):
        super(LightCnnFC1Generator, self).__init__()
        self.gpu_ids = gpu_ids
        self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
        else:
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
        return feature