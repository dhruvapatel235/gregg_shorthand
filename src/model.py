import torch.nn as nn


def _block(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """CNN stroke-feature extractor -> BiGRU -> per-step classifier (CTC).
    Input (B,1,48,256) -> (T=64, B, C) log-probabilities; class 0 is the CTC blank."""

    def __init__(self, num_classes, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(1, 16), nn.MaxPool2d(2, 2),          # 24 x 128
            *_block(16, 32), nn.MaxPool2d(2, 2),         # 12 x 64
            *_block(32, 64), *_block(64, 64), nn.MaxPool2d((2, 1)),      # 6 x 64
            *_block(64, 128), nn.MaxPool2d((2, 1)),      # 3 x 64
            nn.Conv2d(128, 128, (3, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 1 x 64
            nn.Dropout2d(0.1),
        )
        self.rnn = nn.GRU(128, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)  # (T, B, 128)
        out, _ = self.rnn(f)
        return nn.functional.log_softmax(self.fc(out), dim=2)
