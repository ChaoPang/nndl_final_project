from torch import nn, Tensor
from torchvision import models
import torch.nn.functional as F


class FinetuneResnet152(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(FinetuneResnet152, self).__init__()
        model = models.resnet152(pretrained=True)
        self._feature_extractor = nn.Sequential(*(list(model.children())[:-4]))
        # Freeze all the weights
        # for _, p in self._feature_extractor.named_parameters():
        #     p.requires_grad = False
        self._classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self._feature_extractor(x)
        features = F.avg_pool2d(features, features.shape[-1])
        features = features.view(features.size(0), -1)
        out = self._classifier(features)
        return out