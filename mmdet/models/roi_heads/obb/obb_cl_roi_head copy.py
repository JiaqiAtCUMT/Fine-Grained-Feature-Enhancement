from BboxToolkit.vis import bbox
import torch

from mmdet.core import arb2result, arb2roi
from mmdet.models.builder import HEADS

from .obb_standard_roi_head import OBBStandardRoIHead
from mmdet.core.bbox.iou_calculators.obb.obbiou_calculator import OBBOverlaps
@HEADS.register_module()
class OBBCLRoIHead(OBBStandardRoIHead):
    def __init__(self, bbox_roi_extractor=None, bbox_head=None, shared_head=None, train_cfg=None, test_cfg=None):
        super().__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, shared_head=shared_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.iou_calculator = OBBOverlaps()
    """Simplest base roi head including one bbox head and one mask head.
    """
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing"""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_results = self.bbox_head(bbox_feats)
        bbox_results.update(bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training"""
        proposal_bboxes = [res.bboxes for res in sampling_results]
        rois = arb2roi(proposal_bboxes,
                       bbox_type=self.bbox_head.start_bbox_type)
        
        ious,_ = self.iou_calculator(torch.cat(proposal_bboxes,dim=0),torch.cat(gt_bboxes,dim=0)).max(dim=1)
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_targets.update(bbox_results)
        bbox_targets['rois'] = rois
        bbox_targets['ious'] = ious
        gt_labels = gt_labels[:self.bbox_roi_extractor.num_inputs]
        loss_bbox = self.bbox_head.loss(**bbox_targets)

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
