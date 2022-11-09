import torch.nn as nn


class VanillaConvNet(nn.Module):

    def __init__(
            self,
            in_channels,
            num_classes,
            dropout_rate: float = 0.2
    ):
        super().__init__()

        self.block_1 = self._make_simple_conv_layer(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=64,
            num_of_layers=1,
            padding=1,
            is_maxpool=True
        )

        self.block_2 = self._make_simple_conv_layer(
            kernel_size=3,
            in_channels=64,
            out_channels=256,
            num_of_layers=1,
            padding=1,
            is_maxpool=True
        )

        self.block_3 = self._make_simple_conv_layer(
            kernel_size=3,
            in_channels=256,
            out_channels=512,
            num_of_layers=1,
            padding=1,
            is_maxpool=True
        )

        self.block_4 = self._make_simple_conv_layer(
            kernel_size=3,
            in_channels=512,
            out_channels=1024,
            num_of_layers=1,
            padding=1,
            is_maxpool=False
        )

        # Classify
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    @staticmethod
    def _make_simple_conv_layer(
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            num_of_layers: int,
            padding: int = 1,
            is_maxpool: bool = True,
            dropout_rate: float = 0.2
    ):
        layers = []

        for i in range(0, num_of_layers):
            layers.extend(
                [nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False
                ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]
            )

        if is_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2))

        layers.append(nn.Dropout2d(dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layer
        out = self.block_1(x)
        # layer 1
        out = self.block_2(out)
        # layer 2
        out = self.block_3(out)
        # layer 3
        out = self.block_4(out)
        # classification
        out = self.classifier(out)  # 100 x 1024

        return out
