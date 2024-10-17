import torch.nn as nn


"""
copied code for testing
from: https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
"""
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class cifar100_ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128),
                                  conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))
        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028),
                                  conv_block(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1028 x 1 x 1
                                        nn.Flatten(),  # 1028
                                        nn.Linear(1028, num_classes))  # 1028 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out

class cifar100_ResNet9_2(nn.Module): # channel 계산이 안됨...
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128),
                                  conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))

        # self.convPlus1 = conv_block(512, 1024, pool=True)
        # self.convPlus2 = conv_block(1024, 2048, pool=True)
        # self.resPlus1 = nn.Sequential(conv_block(2048, 2048),
        #                               conv_block(2048, 2048))

        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028),
                                  conv_block(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1028 x 1 x 1
                                        nn.Flatten(),  # 1028
                                        nn.Linear(1028, num_classes))  # 1028 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        # out = self.convPlus1(out)
        # out = self.convPlus2(out)
        # out = self.resPlus1(out) + out

        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out






def conv_block_tanh(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.Tanh()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class cifar100_ResNet9_tanh(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()

        self.conv1 = conv_block_tanh(in_channels, 64)
        self.conv2 = conv_block_tanh(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block_tanh(128, 128),
                                  conv_block_tanh(128, 128))

        self.conv3 = conv_block_tanh(128, 256, pool=True)
        self.conv4 = conv_block_tanh(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block_tanh(512, 512),
                                  conv_block_tanh(512, 512))
        self.conv5 = conv_block_tanh(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block_tanh(1028, 1028),
                                  conv_block_tanh(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1028 x 1 x 1
                                        nn.Flatten(),  # 1028
                                        nn.Linear(1028, num_classes))  # 1028 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out

class cifar100_ResNet9_tanh2(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()

        self.conv1 = conv_block_tanh(in_channels, 64)
        self.conv2 = conv_block_tanh(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block_tanh(128, 128),
                                  conv_block_tanh(128, 128))

        self.conv3 = conv_block_tanh(128, 256, pool=True)
        self.conv4 = conv_block_tanh(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block_tanh(512, 512),
                                  conv_block_tanh(512, 512))
        self.conv5 = conv_block_tanh(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block_tanh(1028, 1028),
                                  conv_block_tanh(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1028 x 1 x 1
                                        nn.Flatten(),  # 1028
                                        nn.Linear(1028, 4096),
                                        nn.Linear(4096, num_classes))  # 1028 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out
