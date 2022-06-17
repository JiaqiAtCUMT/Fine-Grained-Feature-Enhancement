import os
import numpy
import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .obb_base import OBBBaseDetector
from .obb_test_mixins import RotateAugRPNTestMixin
from mmdet.cv_core.image.misc import show_tensor,show_obbox
import cv2
def renormalize(tensor_img):
    mean=torch.tensor([123.675, 116.28, 103.53]).reshape(3,1,1)
    std= torch.tensor([58.395, 57.12, 57.375]).reshape(3,1,1)
    return (tensor_img * std + mean).permute(1,2,0).numpy().astype(numpy.uint8)
def show_proposal_obbox(imgs,prorosal_bboxes,step):
    for i,img in enumerate(imgs):
        img = show_obbox(img,prorosal_bboxes[i][:512].cpu(),is_show=False)
        cv2.imwrite(f'visualized/s{step}_proposal_bboxes_{i}.jpg',img)
def show_gt_obbox(imgs,gt_obboxes,gt_labels,step):
    for i,img in enumerate(imgs):
        img = show_obbox(img,gt_obboxes[i].cpu(),labels=gt_labels[i],is_show=False)
        cv2.imwrite(f'visualized/s{step}_gt_bboxes_{i}.jpg',img)
def show_img(imgs,step):
    for i,img in enumerate(imgs):
        cv2.imwrite(f'visualized/s{step}_img_{i}.jpg',img)
def show_heatmap(feats,imgs,method,step):
    for i,feat in enumerate(feats):
        for j in range(feat.shape[0]):
            heat = show_tensor(feat[j],is_show=False,show_split=False,resize_hw=imgs[j].shape[:2],combine_method=method)
            cv2.imwrite(f'visualized/s{step}_{method}_featmap_batch_{j}_head_{i}.jpg',heat)
            heat_map = cv2.addWeighted(imgs[j],0.6,heat,0.4,0)
            cv2.imwrite(f'visualized/s{step}_{method}_feat_batch_{j}_head_{i}.jpg',heat_map)

@DETECTORS.register_module()
class OBBTwoStageDetector(OBBBaseDetector, RotateAugRPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 visualized=False):
        super(OBBTwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)
        self.visualized = visualized
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.step = 0
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBTwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        proposal_type = 'hbb'
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )

        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img.device)
        else:
            # poly proposals need to be generated in roi_head
            proposals = None
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                target_bboxes,
                gt_labels=gt_labels if hasattr(self.rpn_head,'need_label') else None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            if int(os.environ['LOCAL_RANK']) == 0 and self.visualized and self.step%500==0:
                imgs = [renormalize(img.cpu()) for img in img]
                show_proposal_obbox(imgs,proposal_list,self.step)
                show_img(imgs,self.step)
                show_gt_obbox(imgs,target_bboxes,gt_labels,self.step)
                show_heatmap(x,imgs,'sum',self.step)
                show_heatmap(x,imgs,'max',self.step)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_obboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
        losses.update(roi_losses)

        self.step+=1
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if self.visualized and int(os.environ['LOCAL_RANK']) == 0:
            if self.step>=90:
                imgs = [renormalize(img.cpu()) for img in img]
                show_proposal_obbox(imgs,proposal_list,self.step)
                show_heatmap(x,imgs,'sum',self.step)
                show_heatmap(x,imgs,'max',self.step)
                show_img(imgs,self.step)
            if self.step>100:
                exit()
        self.step += 1
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
