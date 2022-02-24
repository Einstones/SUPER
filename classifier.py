import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

x = torch.arange(0, 36).view(1, -1)  # (1, 36)
x = x.repeat(32, 1, 1).float()  # (32, 1, 36)
filter_fw = torch.FloatTensor([[1, 1]]).view(1, 1, 2)
x = F.conv1d(x, filter_fw, padding=1, dilation=2)
