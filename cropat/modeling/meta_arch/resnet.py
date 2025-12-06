import torch
import clip

from detectron2.modeling import build_resnet_backbone
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer


@BACKBONE_REGISTRY.register()
def build_resnet50_backbone(cfg, input_shape):
    assert 'R-50.pkl' in cfg.MODEL.WEIGHTS
    resnet = build_resnet_backbone(cfg, input_shape)
    print('Loading ResNet50 weights')
    DetectionCheckpointer(resnet).load(cfg.MODEL.WEIGHTS)
    return resnet


@BACKBONE_REGISTRY.register()
def build_resnet101_backbone(cfg, input_shape):
    assert 'R-101.pkl' in cfg.MODEL.WEIGHTS
    resnet = build_resnet_backbone(cfg, input_shape)
    print('Loading ResNet101 weights')
    DetectionCheckpointer(resnet).load(cfg.MODEL.WEIGHTS)
    return resnet