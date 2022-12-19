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
    def __init__(
            self,
            img_input_size,
            img_output_size
    ):
        super(ConvAutoEncoder, self).__init__()

        self._img_input_size = img_input_size
        self._img_output_size = img_output_size

        self._encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
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
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._encoder(x)
        out = torch.flatten(out, start_dim=1)
        out = self._encoder_lin(out)

        decoder_out = self._decoder_lin(out)
        decoder_out = self._decoder_unflatten(decoder_out)
        decoder_out = self._decoder(decoder_out)

        return decoder_out


class ConvAutoEncoderV2(nn.Module):
    def __init__(
            self

    ):
        super(ConvAutoEncoderV2, self).__init__()

        self._encoder_layer_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # (16, 8, 8)
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self._encoder_layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1),  # (32, 6, 6)
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self._encoder_layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1),  # (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self._encoder_layer_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1),  # (128, 2, 2)
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self._linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * 2 * 128, 2 * 2 * 128),
            nn.ReLU(True),
            nn.Unflatten(
                dim=1,
                unflattened_size=(128, 2, 2)
            )
        )

        self._decoder_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self._decoder_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1),  # (32, 6, 6)
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self._decoder_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=1),  # (16, 8, 8)
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self._output = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # (3, 32, 32)
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        out1 = self._encoder_layer_1(x)
        out2 = self._encoder_layer_2(out1)
        out3 = self._encoder_layer_3(out2)
        out4 = self._encoder_layer_4(out3)

        decoder_inpout = self._linear(out4)

        decoder_inpout = decoder_inpout + out4

        decoder_out_1 = self._decoder_layer_1(decoder_inpout)
        decoder_out_1 = decoder_out_1 + out3

        decoder_out_2 = self._decoder_layer_2(decoder_out_1)
        decoder_out_2 = decoder_out_2 + out2

        decoder_out_3 = self._decoder_layer_3(decoder_out_2)
        decoder_out_3 = decoder_out_3 + out1
        decoder_out = self._output(decoder_out_3)

        return decoder_out


class SubPixelCNN(nn.Module):
    def __init__(self, upscale_factor=2, channels=3):
        super(SubPixelCNN, self).__init__()
        self._upscale_factor = upscale_factor
        self._channels = channels
        self._sequence = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, channels * (upscale_factor ** 2), 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._sequence(x)
        out = nn.functional.pixel_shuffle(out, upscale_factor=self._upscale_factor)
        return out


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
