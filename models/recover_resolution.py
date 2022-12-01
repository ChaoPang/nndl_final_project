import math
import torch
from torch import nn, Tensor


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


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

        self._encoder_lin = nn.Sequential(
            nn.Linear(2 * 2 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 128)
        )

        self._decoder_lin = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 2 * 2 * 32),
            nn.ReLU(True)
        )

        self._decoder_unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(32, 2, 2)
        )

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._encoder(x)
        out = torch.flatten(out, start_dim=1)
        out = self._encoder_lin(out)

        decoder_out = self._decoder_lin(out)
        decoder_out = self._decoder_unflatten(decoder_out)
        decoder_out = self._decoder(decoder_out)

        return decoder_out


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
