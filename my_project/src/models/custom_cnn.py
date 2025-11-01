import torch.nn as nn


class VolcanoCNN(nn.Module):
    """
    A simple CNN for binary classification:
    non_volcano (class 0) vs volcano (class 1).

    Architecture:
    - Conv(1 -> 32, 3x3) + ReLU
    - MaxPool 2x2
    - Conv(32 -> 64, 3x3) + ReLU
    - MaxPool 2x2
    - Flatten
    - Linear(64 * 8 * 8 -> 128) + ReLU
    - Dropout(0.3)
    - Linear(128 -> 2)

    Assumes input images are resized to 32x32 before being passed in.
    After two 2x2 pools:
        32 -> 16 -> 8
    so feature map is (64, 8, 8), which is 4096 units flattened.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
            # CrossEntropyLoss will apply Softmax implicitly during loss calc.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
