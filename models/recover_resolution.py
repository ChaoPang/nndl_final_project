import torch
import math
from torch import nn, Tensor
from torchvision import models
import torch.nn.functional as F


def make_layer(in_channels, num_of_layers):
    layers = []
    for i in range(0, num_of_layers):
        layers.extend(
            [nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)]
        )

    return nn.Sequential(*layers)


class RecoverHighResolution(nn.Sequential):
    def __init__(
            self,
            in_channels,
            num_of_layers
    ):
        super().__init__(
            make_layer(in_channels, num_of_layers)
        )


def create_recover_resolution_net(
        in_channels=3,
        img_input_size=8,
        img_output_size=32
) -> RecoverHighResolution:
    assert img_output_size % img_input_size == 0
    num_of_layers = math.log(img_output_size // img_input_size, 2)
    if num_of_layers.is_integer():
        return RecoverHighResolution(in_channels, int(num_of_layers))
    raise RuntimeError(f'(img_output_size // img_input_size) needs be base 2 exponent')
