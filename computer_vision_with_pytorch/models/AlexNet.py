import torch.nn as nn
import torch

'''
https://unerue.github.io/computer-vision/classifier-alexnet.html - 모델 코드
https://blog.naver.com/intelliz/221743351934 - 커널 사이즈, 스트라이드 사이즈 구하는 공식
'''

class cifar10_AlexNet1(nn.Module):
    def __init__(self):
        super().__init__()
        # dimension equation = (image_size + 2*padding - kernel_size) / stride + 1
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),nn.ReLU(),
            # (32-4)/2+1 = 15
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (15-3)/2+1 = 7
            nn.Conv2d(32, 128, kernel_size=5, padding=2),nn.ReLU(),
            # (7+2*2-5)/1 + 1 = 7
            nn.MaxPool2d(kernel_size=2, stride=1),
            # (7-2)/1+1 = 6
            nn.Conv2d(128, 196, kernel_size=3, padding=1),nn.ReLU(),
            # (6+2*1-3)/1+1 = 6
            nn.Conv2d(196, 196, kernel_size=3, padding=1),nn.ReLU(),
            # (6+2*1-3)/1+1 = 6
            nn.Conv2d(196, 128, kernel_size=3, padding=1),nn.ReLU(),
            # (6+2*1-3)/1+1 = 6
            nn.MaxPool2d(kernel_size=2, stride=1)
            # (6-2)/1+1 = 5
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

