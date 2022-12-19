from abc import abstractmethod
from enum import Enum

import math
import random

from torch import nn, Tensor
from torchvision import models
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class PretrainedModel(Enum):
    FinetuneEfficientNetV2FeatureExtractor = 'FinetuneEfficientNetV2FeatureExtractor'
    FinetuneRegNetFeatureExtractor = 'FinetuneRegNetFeatureExtractor'
    FinetuneResnet152FeatureExtractor = 'FinetuneResnet152FeatureExtractor'


class PretrainedFeatureExtractor(nn.Module):
    output_num_features = 512

    def __init__(
            self,
            freeze_weight,
            deep_feature: bool = False,
            random_freeze_weight_rate: float = 0.5
    ):
        super(PretrainedFeatureExtractor, self).__init__()
        self._freeze_weight = freeze_weight
        self._deep_feature = deep_feature
        self._random_freeze_weight_rate = random_freeze_weight_rate
        self._feature_extractor = self._get_feature_extractor()

        # Freeze all the weights
        if freeze_weight:
            depth = len(list(self._feature_extractor.features.children()))
            for index, child in enumerate(self._feature_extractor.features.children()):
                prob = math.pow(self._random_freeze_weight_rate, depth - index)
                for _, p in child.named_parameters():
                    if random.random() < prob:
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
    def _get_feature_name(self) -> str:
        pass

    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        return named_features[self._get_feature_name()]


class FinetuneResnet152FeatureExtractor(PretrainedFeatureExtractor):

    def __init__(
            self,
            freeze_weight=False,
            deep_feature=False
    ):
        super(FinetuneResnet152FeatureExtractor, self).__init__(
            deep_feature=deep_feature,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        if self._deep_feature:
            node_name = 'layer3'
        else:
            node_name = 'layer2'
        model = models.resnet152(pretrained=True)
        return create_feature_extractor(
            model,
            return_nodes={node_name: self._get_feature_name()}
        )

    def _get_feature_name(self):
        return 'resnet152_feature'

    def _get_num_features(self) -> nn.Module:
        return list(self._feature_extractor.modules())[-1].num_features


class FinetuneWideResnet101FeatureExtractor(FinetuneResnet152FeatureExtractor):
    def __init__(
            self,
            freeze_weight=False,
            deep_feature=False
    ):
        super(FinetuneResnet152FeatureExtractor, self).__init__(
            deep_feature=deep_feature,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        if self._deep_feature:
            node_name = 'layer3'
        else:
            node_name = 'layer2'
        model = models.wide_resnet101_2(weights='DEFAULT')
        return create_feature_extractor(
            model,
            return_nodes={node_name: self._get_feature_name()}
        )

    def _get_feature_name(self):
        return 'wide_resnet101_feature'


class FinetuneWideResnet101(nn.Sequential):

    def __init__(
            self,
            num_classes,
            dropout_rate=0.5,
            freeze_weight=False,
            deep_feature=False,
            name=None
    ):
        super().__init__(
            FinetuneWideResnet101FeatureExtractor(
                deep_feature=deep_feature,
                freeze_weight=freeze_weight
            ),
            *create_head_classifier(num_classes, dropout_rate)
        )
        self._name = name if name else 'FinetuneWideResnet101'

    @property
    def name(self):
        return self._name


class FinetuneResnet152(nn.Sequential):

    def __init__(
            self,
            num_classes,
            dropout_rate=0.5,
            freeze_weight=False,
            deep_feature=False,
            name=None
    ):
        super().__init__(
            FinetuneResnet152FeatureExtractor(
                deep_feature=deep_feature,
                freeze_weight=freeze_weight
            ),
            *create_head_classifier(num_classes, dropout_rate)
        )
        self._name = name if name else 'FinetuneResnet152'

    @property
    def name(self):
        return self._name


def create_head_classifier(
        num_classes,
        dropout_rate
):
    return [
        nn.Linear(FinetuneRegNetFeatureExtractor.output_num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(
            256,
            num_classes
        )]


class FinetuneRegNetFeatureExtractor(PretrainedFeatureExtractor):
    def __init__(
            self,
            freeze_weight=False,
            deep_feature=False
    ):
        super(FinetuneRegNetFeatureExtractor, self).__init__(
            deep_feature=deep_feature,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:

        # Extract deep features if the image size is bigger
        if self._deep_feature:
            node_name = 'trunk_output.block3'
        else:
            node_name = 'trunk_output.block2'

        model = models.regnet_y_128gf(weights='DEFAULT')
        return create_feature_extractor(
            model,
            return_nodes={node_name: self._get_feature_name()}
        )

    def _get_feature_name(self):
        return 'regnet_feature'

    def _get_tensor_by_name(self, named_features: dict) -> Tensor:
        return named_features[self._get_feature_name()]

    def _get_num_features(self) -> nn.Module:
        # The last the layer is RELU so we need to go to the second last layer to infer the shape
        return list(self._feature_extractor.modules())[-2].num_features


class FinetuneRegNet(nn.Sequential):

    def __init__(
            self,
            num_classes,
            dropout_rate=0.5,
            freeze_weight=False,
            deep_feature=False,
            name=None
    ):
        super().__init__(
            FinetuneRegNetFeatureExtractor(
                deep_feature=deep_feature,
                freeze_weight=freeze_weight
            ),
            *create_head_classifier(num_classes, dropout_rate)
        )
        self._name = name if name else 'FinetuneRegNet'

    @property
    def name(self):
        return self._name


class FinetuneEfficientNetV2FeatureExtractor(PretrainedFeatureExtractor):

    def __init__(
            self,
            freeze_weight=False,
            deep_feature=False
    ):
        super(FinetuneEfficientNetV2FeatureExtractor, self).__init__(
            deep_feature=deep_feature,
            freeze_weight=freeze_weight
        )

    def _get_feature_extractor(self) -> nn.Module:
        # Extract deep features if the image size is bigger
        if self._deep_feature:
            node_name = 'features.4.9.block.3'
        else:
            node_name = 'features.3.6.block.1'

        return create_feature_extractor(
            self._get_pretrained_model(),
            return_nodes={node_name: self._get_feature_name()}
        )

    def _get_feature_name(self) -> str:
        return 'eff_net_v2'

    def _get_num_features(self) -> nn.Module:
        return list(self._feature_extractor.modules())[-1].num_features

    def _get_pretrained_model(self) -> nn.Module:
        return models.efficientnet_v2_l(
            weights=models.efficientnet.EfficientNet_V2_L_Weights
        )


class FinetuneEfficientNetB7FeatureExtractor(FinetuneEfficientNetV2FeatureExtractor):

    def __init__(
            self,
            freeze_weight=False,
            deep_feature=False
    ):
        super(FinetuneEfficientNetV2FeatureExtractor, self).__init__(
            deep_feature=deep_feature,
            freeze_weight=freeze_weight
        )

    def _get_feature_name(self) -> str:
        return 'eff_net_b7'

    def _get_pretrained_model(self) -> nn.Module:
        return models.efficientnet_b7(
            weights=models.efficientnet.EfficientNet_B7_Weights
        )

    def _get_num_features(self) -> nn.Module:
        return list(self._feature_extractor.modules())[-2].num_features


class FinetuneEfficientNetV2(nn.Sequential):

    def __init__(
            self,
            num_classes,
            dropout_rate=0.5,
            freeze_weight=False,
            deep_feature=False,
            name=None
    ):
        super().__init__(
            FinetuneEfficientNetV2FeatureExtractor(
                deep_feature=deep_feature,
                freeze_weight=freeze_weight
            ),
            *create_head_classifier(num_classes, dropout_rate)
        )
        self._name = name if name else 'FinetuneEfficientNetV2'

    @property
    def name(self):
        return self._name


def create_multitask_trainer(
        num_classes,
        num_subclasses,
        pretrained_model: PretrainedModel,
        dropout_rate=0.5,
        freeze_weight=False,
        deep_feature=False,
        name=None
) -> nn.Module:
    if pretrained_model == PretrainedModel.FinetuneEfficientNetV2FeatureExtractor:
        finetune_extractor_class = FinetuneEfficientNetV2FeatureExtractor
    elif pretrained_model == PretrainedModel.FinetuneRegNetFeatureExtractor:
        finetune_extractor_class = FinetuneRegNetFeatureExtractor
    elif pretrained_model == PretrainedModel.FinetuneResnet152FeatureExtractor:
        finetune_extractor_class = FinetuneResnet152FeatureExtractor
    else:
        raise RuntimeError(
            f'We only support FinetuneEfficientNetV2FeatureExtractor,  '
            f'FinetuneRegNetFeatureExtractor, and FinetuneResnet152FeatureExtractor')

    return FinetuneWithMultiTask(
        num_classes=num_classes,
        num_subclasses=num_subclasses,
        finetune_extractor_class=finetune_extractor_class,
        dropout_rate=dropout_rate,
        freeze_weight=freeze_weight,
        deep_feature=deep_feature,
        name=name
    )


class FinetuneWithMultiTask(nn.Module):

    def __init__(
            self,
            num_classes: int,
            num_subclasses: int,
            finetune_extractor_class: PretrainedFeatureExtractor,
            dropout_rate: float = 0.5,
            freeze_weight: bool = False,
            deep_feature: bool = False,
            name: str = None
    ):
        super(FinetuneWithMultiTask, self).__init__(

        )
        self._feature_extractor = finetune_extractor_class(
            freeze_weight=freeze_weight,
            deep_feature=deep_feature
        )
        self._super_classifier = nn.Sequential(
            *create_head_classifier(num_classes, dropout_rate))
        self._subclass_classifier = nn.Sequential(
            *create_head_classifier(num_subclasses, dropout_rate))
        self._name = name if name else finetune_extractor_class.__name__

    def forward(self, x: Tensor) -> Tensor:
        # This returns a named feature
        features = self._feature_extractor(x)
        subclass_out = self._subclass_classifier(
            features
        )
        superclass_out = self._super_classifier(
            features
        )
        return superclass_out, subclass_out

    @property
    def name(self):
        return self._name


class FinetuneEfficientNetB7(nn.Sequential):

    def __init__(
            self,
            num_classes,
            dropout_rate=0.5,
            freeze_weight=False,
            deep_feature=False,
            name=None
    ):
        super().__init__(
            FinetuneEfficientNetB7FeatureExtractor(
                deep_feature=deep_feature,
                freeze_weight=freeze_weight
            ),
            *create_head_classifier(num_classes, dropout_rate)
        )
        self._name = name if name else 'FinetuneEfficientNetB7'

    @property
    def name(self):
        return self._name
