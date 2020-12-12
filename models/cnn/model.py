import torch
from torch import nn
from torch.nn import functional as F


class SeqCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.encoder = Encoder(input_size)
        self.fd = self.get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.fd, self.fd),
            nn.ReLU()
        )
        self.top_layer = None

    def get_feature_dim(self):
        inp = torch.zeros(1, self.input_size)
        out = self.encoder(inp)
        fd = out.shape[1]
        return fd

    def forward(self, x):
        out = self.encoder(x)
        out = self.classifier(out)
        if self.top_layer is not None:
            out = self.top_layer(F.relu(out))
        return out


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.encoder_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=512, kernel_size = 3, stride = 2, padding = 1),
                # nn.Linear(),
                # nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size = 3, stride = 2, padding = 1),
                # nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size = 3, stride = 2, padding = 1),
                # nn.BatchNorm1d(n)
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm1d(32),
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size = 3, stride = 2, padding = 1)
                # nn.Linear(in_features=15, out_features=n)
                )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encoder_conv(x)
        out = out.reshape(out.shape[0], -1)
        return out
        