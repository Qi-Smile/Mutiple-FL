from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn
from torchvision import models as tv_models

from .simple import LeNet


_WEIGHTS_MAP = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "vgg16": "VGG16_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
}


def _resolve_weights_kwargs(arch: str, pretrained: bool) -> Dict:
    if not pretrained:
        return {}
    weights_attr = _WEIGHTS_MAP.get(arch)
    if weights_attr:
        weights_enum = getattr(tv_models, weights_attr, None)
        if weights_enum is not None:
            return {"weights": weights_enum.DEFAULT}
    # Fall back to legacy flag if running on older torchvision
    return {"pretrained": True}


def _replace_first_conv(conv_layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv_layer.in_channels == in_channels:
        return conv_layer
    bias = conv_layer.bias is not None
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=bias,
        padding_mode=conv_layer.padding_mode,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if bias:
        nn.init.zeros_(new_conv.bias)
    return new_conv


def get_model_builder(name: str, num_classes: int, **kwargs) -> Callable[[], nn.Module]:
    """Return a callable that builds a model by name."""
    name = name.lower()
    if name == "lenet":
        in_channels = kwargs.get("in_channels", 1)
        return lambda: LeNet(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18":
        arch = "resnet18"
    elif name == "resnet34":
        arch = "resnet34"
    elif name == "resnet50":
        arch = "resnet50"
    else:
        arch = None

    if arch and arch.startswith("resnet"):
        in_channels = kwargs.get("in_channels", 3)
        pretrained = kwargs.get("pretrained", False)

        def build_resnet() -> nn.Module:
            weight_kwargs = _resolve_weights_kwargs(arch, pretrained)
            model = getattr(tv_models, arch)(**weight_kwargs)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            if in_channels != model.conv1.in_channels:
                model.conv1 = _replace_first_conv(model.conv1, in_channels)
            return model

        return build_resnet

    if name == "vgg16":
        in_channels = kwargs.get("in_channels", 3)
        pretrained = kwargs.get("pretrained", False)

        def build_vgg() -> nn.Module:
            weight_kwargs = _resolve_weights_kwargs("vgg16", pretrained)
            model = tv_models.vgg16(**weight_kwargs)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            if in_channels != model.features[0].in_channels:
                model.features[0] = _replace_first_conv(model.features[0], in_channels)
            return model

        return build_vgg

    if name == "mobilenet_v2":
        in_channels = kwargs.get("in_channels", 3)
        pretrained = kwargs.get("pretrained", False)

        def build_mobilenet() -> nn.Module:
            weight_kwargs = _resolve_weights_kwargs("mobilenet_v2", pretrained)
            model = tv_models.mobilenet_v2(**weight_kwargs)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            first_conv = model.features[0][0]
            if in_channels != first_conv.in_channels:
                model.features[0][0] = _replace_first_conv(first_conv, in_channels)
            return model

        return build_mobilenet

    if name == "efficientnet_b0":
        in_channels = kwargs.get("in_channels", 3)
        pretrained = kwargs.get("pretrained", False)

        def build_efficientnet() -> nn.Module:
            weight_kwargs = _resolve_weights_kwargs("efficientnet_b0", pretrained)
            model = tv_models.efficientnet_b0(**weight_kwargs)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            first_conv = model.features[0][0]
            if in_channels != first_conv.in_channels:
                model.features[0][0] = _replace_first_conv(first_conv, in_channels)
            return model

        return build_efficientnet

    raise ValueError(f"Model '{name}' is not supported by default.")


def list_available_models() -> Dict[str, str]:
    return {
        "lenet": "LeNet-style CNN supporting variable input channels.",
        "resnet18": "ResNet-18 backbone with configurable input channels.",
        "resnet34": "ResNet-34 backbone for deeper residual learning.",
        "resnet50": "ResNet-50 backbone with bottleneck blocks.",
        "vgg16": "VGG-16 classifier with fully-connected head replacement.",
        "mobilenet_v2": "MobileNetV2 lightweight architecture for mobile/edge.",
        "efficientnet_b0": "EfficientNet-B0 compound scaled CNN.",
    }
