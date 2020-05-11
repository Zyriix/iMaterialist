# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import SigmoidFocalLoss
import numpy as np
import json
# from 
class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        f= open('../configs/pos_weight.json','r')
        self.weight = torch.Tensor(json.load(f)).to(device)
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.sigmoid_focal_loss = SigmoidFocalLoss(0,0.5)
        self.ATTRIBUTES_ON = cfg.MODEL.ROI_BOX_HEAD.ATTIBUTES_ON
        

    def match_targets_to_proposals(self, proposal, target):
        # print(f"before match,{target.get_field('categories').shape}")
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        # TODO: 检查维度是否一致
        target = target.copy_with_fields(["labels","categories"])
        # target = target.copy_with_fields()
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # print(f"after match,{target.get_field('categories').shape}")
        # print(matched_idxs.clamp(min=0))
        # print(target.get_field("categories").shape)
        matched_targets = target[matched_idxs.clamp(min=0)]
        # print(matched_targets.get_field("categories").shape)
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        categories  = []
        
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            
                # print(cats_per_image.shape)

            

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
            

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

            
                
            cats_per_image = matched_targets.get_field("categories")
            # print(cats_per_image.shape)
            cats_per_image = cats_per_image.to(dtype=torch.int64)
            cats_per_image[bg_inds] = torch.Tensor(np.zeros(294)).cuda().long()
            cats_per_image[ignore_inds] = -1
            
            categories.append(cats_per_image)

        return labels,categories, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, categories,regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image,cat_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels,categories, regression_targets, proposals
        ):
            # print(f"before subsample:{cat_per_image.shape}")
            proposals_per_image.add_field("labels", labels_per_image)
            if self.ATTRIBUTES_ON:
                proposals_per_image.add_field("categories", cat_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        
        return proposals

    def __call__(self, class_logits, cat_logits,box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            cat_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        cat_logits = cat(cat_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        # print(cat_logits,torch.sum(categories))

        classification_loss = F.cross_entropy(class_logits, labels)*2
        # test = F.cross_entropy(class_logits, labels)
       
        # classification_loss=torch.sum(classification_loss)/classification_loss.shape[0]
        # print(classification_loss,test)
        # print(cat_logits.shape,categories.shape)
        # print(cat_logits,categories)
        # print(torch.sum(categories,axis=0))
        # categories_loss = self.sigmoid_focal_loss(cat_logits,categories.int())
        
        # categories_loss = F.binary_cross_entropy_with_logits(cat_logits,categories.float(),pos_weight=self.weight)
        categories_loss=0
        if self.ATTRIBUTES_ON:
            # print(labels.shape)

            """
            new version's loss computation, with categories's shape [batch,num_categories*num_classes]
            """
            positive_inds = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[positive_inds]
            categories = cat([proposal.get_field("categories") for proposal in proposals], dim=0)[positive_inds]
            # print()
            # indices = np.zeros(294*47)
            # indices[(labels_pos*294).int():((labels_pos+1)*294).int()]=1
            cat_logits_pos = torch.zeros_like(categories).float()
            for i,(indx,label) in enumerate(zip(positive_inds,labels_pos)):
                cat_logits_pos[i]=(cat_logits[indx,label*294:(label+1)*294])
                # print(cat_logits_pos)
            
            categories_loss = F.binary_cross_entropy_with_logits(cat_logits_pos,categories.float())*10
            # focal_loss = self.sigmoid_focal_loss(cat_logits_pos,categories.float())
            # print(focal_loss)
            
            """
            old version's loss computation, with categories's shape [batch,num_categories]
            """
            # categories_loss = F.binary_cross_entropy_with_logits(cat_logits,categories.float(),reduction='none')
        # cat_loss = F.binary_cross_entropy_with_logits(cat_logits,categories.float())
            # categories_loss=torch.sum(categories_loss)/labels_pos.numel()/2
            # print(classification_loss,categories_loss)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss,categories_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        cfg,
        matcher,
        fg_bg_sampler, 
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
