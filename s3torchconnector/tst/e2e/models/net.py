import torch
import torch.nn.functional as F
from torch import nn

# Based on https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def equals(self, other_model: nn.Module) -> bool:
        for key_item_1, key_item_2 in zip(
            self.state_dict().items(), other_model.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                return False
        return True
