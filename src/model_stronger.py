import torch.nn as nn
import torch.nn.functional as F


def _block(i, o):
    return [
        nn.Conv2d(i, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    ]


class CRNN(nn.Module):
    """Stronger CNN-BiGRU-CTC recognizer.

    Input: (B, 1, 48, 256)
    CNN output: (B, 128, 1, 64)
    RNN input: 64 time steps
    """

    def __init__(self, num_classes, hidden=128):
        super().__init__()

        self.cnn = nn.Sequential(
            *_block(1, 32),
            nn.MaxPool2d(2, 2),              # 24 x 128

            *_block(32, 64),
            nn.MaxPool2d(2, 2),              # 12 x 64

            *_block(64, 128),
            *_block(128, 128),
            nn.MaxPool2d((2, 1)),            # 6 x 64

            *_block(128, 128),
            nn.MaxPool2d((2, 1)),            # 3 x 64

            nn.Conv2d(128, 128, (3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),           # 1 x 64

            nn.Dropout2d(0.1),
        )

        self.rnn = nn.GRU(
            128,
            hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.rnn(f)
        return F.log_softmax(self.fc(out), dim=2)