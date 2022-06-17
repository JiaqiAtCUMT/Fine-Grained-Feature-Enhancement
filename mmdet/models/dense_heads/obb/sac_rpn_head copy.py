from numpy.lib.polynomial import poly
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import numpy as np
import cv2
from mmdet.ops import arb_batched_nms
from mmdet.core import obb2hbb,multi_apply,obb2poly
from mmdet.models.builder import HEADS
from .obb_anchor_head import OBBAnchorHead
from ..rpn_test_mixin import RPNTestMixin
from mmcv.cnn import ConvModule, xavier_init

def map_obb_levels(obboxes, num_levels, finest_scale = 56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = torch.sqrt(obboxes[:, 2] * obboxes[:, 3])
    target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls

class SAC(nn.Module):
    def __init__(self,feat_channels):
        super().__init__()
        self.feat_channels = feat_channels
        self.channel_attn_gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn_fc1 = nn.Linear(self.feat_channels,self.feat_channels)
        self.channel_attn_fc2 = nn.Linear(self.feat_channels,self.feat_channels)
        
        self.pix_attn_convs = nn.ModuleList([
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                1,stride=1,padding=0,
                norm_cfg=dict(type='BN', requires_grad=True),
                inplace=False),
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                3,stride=1,padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                inplace=False),
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                3,stride=1,padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                inplace=False),
            ConvModule(
                self.feat_channels,
                1,
                3,stride=1,padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                inplace=False)
        ])
        self.bce_loss = nn.BCEWithLogitsLoss()

        normal_init(self.channel_attn_fc1, std=0.01)
        normal_init(self.channel_attn_fc2, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self,features):
        pix_attn = features
        for conv in self.pix_attn_convs:
            pix_attn = conv(pix_attn)
        pix_attn = torch.sigmoid(pix_attn)
        
        channel_attn = self.channel_attn_gap(features).reshape(-1,self.feat_channels)
        channel_attn = self.channel_attn_fc1(channel_attn)
        channel_attn = torch.relu(channel_attn)
        channel_attn = self.channel_attn_fc2(channel_attn)
        channel_attn = torch.sigmoid(channel_attn).reshape(-1,self.feat_channels,1,1)

        out = pix_attn * channel_attn * features
        return out, pix_attn
    def loss(self,pix_attn,target_obboxes):
        return self.bce_loss(pix_attn,target_obboxes)
@HEADS.register_module()
class SACRPNHead(RPNTestMixin, OBBAnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels,img_size = 1024, **kwargs):
        super(SACRPNHead, self).__init__(
            1,
            in_channels,
            bbox_type='obb',
            reg_dim=6,
            background_label=0,
            **kwargs)
        self.img_size = img_size
    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        self.sac = SAC(self.feat_channels)
    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x, pixel_attn = self.sac(x)
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)

        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred, pixel_attn

    def loss(self,
             cls_scores,
             bbox_preds,
             pixel_attns,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(SACRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        masks = []
        for i in range(len(gt_bboxes)):
            polys = obb2poly(gt_bboxes[i]).cpu().numpy().astype(np.int).reshape((-1,4,2))
            mask = np.zeros((self.img_size,self.img_size,1))
            cv2.fillPoly(mask,polys,255)
            mask = mask.reshape(1,1,self.img_size,self.img_size)
            mask = torch.tensor(mask / 255.,dtype=torch.float)
            masks.append(mask)
        masks = torch.cat(masks,dim=0)
        masks = masks.to(cls_scores[0].device)
        multi_mask = []
        for i in range(len(pixel_attns)):
            h,w = pixel_attns[i].shape[2:]
            multi_mask.append(F.interpolate(masks,(h,w)))
        loss_sac = [self.sac.loss(pixel_attn,mask) for pixel_attn,mask in zip(pixel_attns,multi_mask)]
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'],loss_sac=loss_sac)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_dim)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w, h = proposals[:, 2], proposals[:, 3]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        hproposals = obb2hbb(proposals)
        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        _, keep = arb_batched_nms(hproposals, scores, ids, nms_cfg)

        dets = torch.cat([proposals, scores[:, None]], dim=1)
        dets = dets[keep]
        return dets[:cfg.nms_post]

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs[:2], img_metas)
        return proposal_list