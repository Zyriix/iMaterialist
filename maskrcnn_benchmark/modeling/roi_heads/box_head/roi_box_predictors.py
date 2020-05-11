# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_categories = cfg.MODEL.ROI_BOX_HEAD.NUM_CATEGORIES
        representation_size = in_channels
        self.ATTIBUTES_ON = cfg.MODEL.ROI_BOX_HEAD.ATTIBUTES_ON
        
        

        self.cls_score = nn.Linear(representation_size, num_classes) 

        # one predictor
        

        # self.cat_score =[ nn.Linear(representation_size, num_categories)for _ in range(num_classes)]
        
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score,self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.ATTIBUTES_ON:
            self.cat_score=nn.Linear(representation_size, num_categories*num_classes)
            """old version"""
            # self.cat_score=nn.Linear(representation_size+num_classes, num_categories)
            nn.init.normal_(self.cat_score.weight, std=0.01)
            nn.init.constant_(self.cat_score.bias, 0)
            
            
            

        # for x in self.cat_score:
        #   nn.init.normal_(x.weight, std=0.01)
        #   nn.init.constant_(x.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        # print(scores.shape,x.shape)

        
        # one predictor
        cat_scores = None
        if self.ATTIBUTES_ON:
            # labels = torch.max(scores,1)[1]
            cat_scores = self.cat_score(x)
            """old version"""
            # cat_scores = self.cat_score(torch.cat((x,scores),1))

        bbox_deltas = self.bbox_pred(x)

        return scores,cat_scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
