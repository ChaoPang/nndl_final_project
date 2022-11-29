from torch import nn, Tensor
from torchvision import models
import torch.nn.functional as F


class FinetuneResnet152(nn.Module):
    def __init__(self, num_classes):
        super(FinetuneResnet152, self).__init__()
        model = models.resnet152(pretrained=True)
        self._feature_extractor = nn.Sequential(*(list(model.children())[:-5]))
        # Freeze all the weights
        for _, p in self._feature_extractor.named_parameters():
            p.requires_grad = False
        self._classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self._feature_extractor(x)
        features = F.avg_pool2d(features, features.shape[-1])
        features = features.view(features.size(0), -1)
        out = self._classifier(features)
        return out
