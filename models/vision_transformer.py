import torch
from torch import nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 2,
            embedding_size: int = 64,
            img_size: int = 8
    ):
        super(PatchEmbedding, self).__init__()
        self._in_channels = in_channels
        self._patch_size = patch_size
        self._embedding_size = embedding_size

        self._cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_size)
        )
        # self._positions = nn.Parameter(
        #     torch.randn(1, (img_size // patch_size) ** 2 + 1, embedding_size))
        #
        self._positions = self._create_positional_encodings()

        self._conv2d_layer = nn.Conv2d(
            in_channels,
            embedding_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, _, _, _ = x.shape
        x = self._conv2d_layer(x)
        x = x.view((bs, -1, self._embedding_size))
        x = torch.cat([self._cls_token.repeat((bs, 1, 1)), x], dim=1)

        bs, seq_len, _ = x.shape
        # Add positional encodings to each position
        positional_encodings = self._positions[:seq_len].unsqueeze(0)
        x += positional_encodings

        return x

    def _create_positional_encodings(
            self,
            max_seq_len=1000
    ):
        """Creates positional encodings for the inputs.

        Arguments:
            max_seq_len: a number larger than the maximum string length we expect to encounter during training

        Returns:
            pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len.
        """
        pos_indices = torch.arange(max_seq_len)[..., None]
        dim_indices = torch.arange(self._embedding_size // 2)[None, ...]
        exponents = (2 * dim_indices).float() / self._embedding_size
        trig_args = pos_indices / (10000 ** exponents)
        sin_terms = torch.sin(trig_args)
        cos_terms = torch.cos(trig_args)

        pos_encodings = torch.zeros((max_seq_len, self._embedding_size))
        pos_encodings[:, 0::2] = sin_terms
        pos_encodings[:, 1::2] = cos_terms

        if torch.cuda.is_available():
            pos_encodings = pos_encodings.cuda()

        return pos_encodings


class VisionTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            img_size: int = 8,
            patch_size: int = 2,
            embedding_size: int = 128,
            num_heads: int = 8,
            num_layers=6,
            n_classes: int = 3
    ):
        super(VisionTransformer, self).__init__()

        self._patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            embedding_size=embedding_size
        )
        self._transformer_encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads
            ),
            num_layers=num_layers
        )

        self._classifier = nn.Linear(embedding_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self._patch_embedding(x)
        x = self._transformer_encoder_layer(x)
        x = self._classifier(torch.squeeze(x[:, 0]))
        return x
