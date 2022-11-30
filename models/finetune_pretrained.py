from abc import abstractmethod

import torch
from torch import nn, Tensor
from torchvision import models
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class PretrainedFeatureExtractor(nn.Module):
    output_num_features = 256

    def __init__(self, freeze_weight, dropout_rate):
        super(PretrainedFeatureExtractor, self).__init__()
        self._freeze_weight = freeze_weight
        self._feature_extractor = self._get_feature_extractor()

        # Freeze all the weights
        if freeze_weight:
            for _, p in self._feature_extractor.named_parameters():
                p.requires_grad = False

        self._linear_layer = nn.Sequential(
            nn.Linear(self._get_num_features(), self.output_num_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        # This returns a named feature
        named_features = self._feature_extractor(x)
        # Extract the tensor
        features = self._get_tensor_by_name(named_features)
        features = F.avg_pool2d(features, features.shape[-1])
        features = features.view(features.size(0), -1)
        out = self._linear_layer(features)
        return out

    @abstractmethod
    def _get_num_features(self) -> nn.Module:
        pass

    @abstractmethod
    def _get_feature_extractor(self) -> nn.Module:
        pass

    @abstractmethod
    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        pass


class FinetuneResnet152FeatureExtractor(PretrainedFeatureExtractor):

    def __init__(self, dropout_rate=0.5, freeze_weight=False):
        super(FinetuneResnet152FeatureExtractor, self).__init__(
            dropout_rate=dropout_rate,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        model = models.resnet152(pretrained=True)
        return create_feature_extractor(
            model,
            return_nodes={'layer2': 'layer2'}
        )

    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        return named_features['layer2']

    def _get_num_features(self) -> nn.Module:
        return list(self._feature_extractor.modules())[-1].num_features


class FinetuneResnet152(nn.Sequential):

    def __init__(self, num_classes, dropout_rate=0.5, freeze_weight=False):
        super().__init__(
            FinetuneResnet152FeatureExtractor(
                dropout_rate=dropout_rate,
                freeze_weight=freeze_weight
            ),
            nn.Linear(FinetuneResnet152FeatureExtractor.output_num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                64,
                num_classes
            )
        )


class FinetuneRegNetFeatureExtractor(PretrainedFeatureExtractor):
    def __init__(self, dropout_rate=0.5, freeze_weight=False):
        super(FinetuneRegNetFeatureExtractor, self).__init__(
            dropout_rate=dropout_rate,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        model = models.regnet_y_32gf(pretrained=True)
        return create_feature_extractor(
            model,
            return_nodes={'trunk_output.block2': 'block2'}
        )

    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        return named_features['block2']

    def _get_num_features(self) -> nn.Module:
        # The last the layer is RELU so we need to go to the second last layer to infer the shape
        return list(self._feature_extractor.modules())[-2].num_features


class FinetuneRegNet(nn.Sequential):

    def __init__(self, num_classes, dropout_rate=0.5, freeze_weight=False):
        super().__init__(
            FinetuneRegNetFeatureExtractor(
                dropout_rate=dropout_rate,
                freeze_weight=freeze_weight
            ),
            nn.Linear(FinetuneResnet152FeatureExtractor.output_num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                64,
                num_classes
            )
        )


class FinetuneEfficientNetB7FeatureExtractor(PretrainedFeatureExtractor):

    def __init__(self, dropout_rate=0.5, freeze_weight=False):
        super(FinetuneEfficientNetB7FeatureExtractor, self).__init__(
            dropout_rate=dropout_rate,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        model = models.efficientnet_b7(pretrained=True)
        return create_feature_extractor(
            model,
            return_nodes={'features.3': 'features.3'}
        )

    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        return named_features['features.3']

    def _get_num_features(self) -> nn.Module:
        return list(self._feature_extractor.modules())[-1].num_features


class FinetuneEfficientNetB7(nn.Sequential):

    def __init__(self, num_classes, dropout_rate=0.5, freeze_weight=False):
        super().__init__(
            FinetuneEfficientNetB7FeatureExtractor(
                dropout_rate=dropout_rate,
                freeze_weight=freeze_weight
            ),
            nn.Linear(FinetuneEfficientNetB7FeatureExtractor.output_num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                64,
                num_classes
            )
        )


class FinetuneEnsembleModel(nn.Module):
    def __init__(self, num_classes, device, dropout_rate=0.5, freeze_weight=False):
        super(FinetuneEnsembleModel, self).__init__()
        self._feature_extractors = [
            FinetuneEfficientNetB7FeatureExtractor(
                dropout_rate=dropout_rate,
                freeze_weight=freeze_weight
            ).to(device),
            FinetuneResnet152FeatureExtractor(
                dropout_rate=dropout_rate,
                freeze_weight=freeze_weight
            ).to(device)
            # FinetuneRegNetFeatureExtractor(
            #     dropout_rate=dropout_rate,
            #     freeze_weight=freeze_weight
            # ).to(device)
        ]

        num_of_features = PretrainedFeatureExtractor.output_num_features * len(
            self._feature_extractors)

        self._classifier = nn.Sequential(
            nn.Linear(num_of_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        ensemble_features = []
        for feature_extractor in self._feature_extractors:
            ensemble_features.append(feature_extractor(x))
        ensemble_features = torch.concat(ensemble_features, dim=-1)
        out = self._classifier(ensemble_features)
        return out
