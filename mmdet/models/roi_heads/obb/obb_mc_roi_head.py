import torch

from mmdet.core import arb2result, arb2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head, build_roi_extractor

from .obb_test_mixins import OBBoxTestMixin
from .obb_base_roi_head import OBBBaseRoIHead
from .obb_standard_roi_head import OBBStandardRoIHead

@HEADS.register_module()
class OBBMCRoIHead(OBBStandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head.
    """
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing"""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training"""
        rois = arb2roi([res.bboxes for res in sampling_results],
                       bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = arb2result(det_bboxes, det_labels, self.bbox_head.num_classes,
                                  bbox_type=self.bbox_head.end_bbox_type)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes = []
        det_labels = []
        for cls_score,bbox_pred in zip(bbox_results['cls_score'],bbox_results['bbox_pred']):
            result = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            
            det_bboxes.append(result[0])
            det_labels.append(result[1])
        det_bboxes = torch.cat(det_bboxes,dim=0)
        det_labels = torch.cat(det_labels,dim=0)
        return det_bboxes, det_labels

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        # proposal_list,pred_index =zip(*proposal_list)
        # pred_inds = []
        if self.with_bbox:
            start_bbox_type = self.bbox_head.start_bbox_type
            end_bbox_type = self.bbox_head.end_bbox_type
            target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore \
                    if start_bbox_type == 'hbb' else gt_obboxes_ignore

            num_imgs = len(img_metas)
            if target_bboxes_ignore is None:
                target_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], target_bboxes[i],
                    target_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    target_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if start_bbox_type != end_bbox_type:
                    if gt_obboxes[i].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeors(
                            (0, gt_obboxes[0].size(-1)))
                    else:
                        sampling_result.pos_gt_bboxes = \
                                gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)
                # pred_inds.append([sampling_result.pos_gt_labels, pred_index[i][sampling_result.neg_inds-sampling_result.num_gts]])

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,# pred_inds,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses