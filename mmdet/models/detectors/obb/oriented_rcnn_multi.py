from mmdet.models.builder import DETECTORS
from .obb_two_stage_multi import OBBTwoStageDetectorMulti


@DETECTORS.register_module()
class OrientedRCNNMulti(OBBTwoStageDetectorMulti):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OrientedRCNNMulti, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
