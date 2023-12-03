from torch import nn
import torch.nn.functional as F

class CIFARModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.convolutional_layer1 = nn.Conv2d(3, 9, 3)
        self.convolutional_layer2 = nn.Conv2d(9, 18, 4)
        self.linear_layer1 = nn.Linear(648, 300)
        self.linear_layer2 = nn.Linear(300, 10)
        self.linear_layer3 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 648)
        x = self.linear_layer1(x)
        x = F.relu(x)
        x = self.linear_layer2(x)
        x = F.relu(x)
        self.linear_layer3(x)
        x = F.relu(x)
        F.max_pool2d(x)

        return x

criterion = nn.CrossEntropyLoss()