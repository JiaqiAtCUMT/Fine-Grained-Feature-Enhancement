import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .obbox_head import OBBoxHead


@HEADS.register_module()
class BranchBBoxHead(OBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(BranchBBoxHead, self).__init__(*args, **kwargs)
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cls_convs = nn.ModuleList([
            ConvModule(
                self.in_channels,
                self.conv_out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
            ConvModule(
                self.in_channels,
                self.conv_out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        ])
        self.reg_convs = nn.ModuleList([
            ConvModule(
                self.in_channels,
                self.conv_out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
            ConvModule(
                self.in_channels,
                self.conv_out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        ])
        self.cls_gap = nn.AdaptiveAvgPool2d(3)
        self.cls_gmp = nn.AdaptiveMaxPool2d(3)

        self.reg_gap = nn.AdaptiveAvgPool1d(1)
        self.reg_gmp = nn.AdaptiveMaxPool1d(1)
        self.cls_sec_conv = ConvModule(
                self.conv_out_channels*2,
                self.conv_out_channels,
                3,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
                    
        self.fc_cls = nn.Linear(conv_out_channels, self.num_classes + 1)
        out_dim_reg = self.reg_dim if self.reg_class_agnostic else \
                    self.reg_dim * self.num_classes
        self.fc_reg = nn.Linear(self.roi_feat_area*2, out_dim_reg)
    def init_weights(self):
        super(BranchBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.cls_convs, self.reg_convs, self.cls_sec_conv]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B,C,H,W = x.shape
        x_cls = x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        x_cls = torch.cat([self.cls_gap(x_cls),self.cls_gmp(x_cls)],dim=1)
        x_cls = self.cls_sec_conv(x_cls).reshape(B,C)

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        x_reg = x_reg.reshape(B,C,H*W).permute(0,2,1)
        x_reg = torch.cat([self.reg_gap(x_reg),self.reg_gmp(x_reg)],dim=1).reshape(B,H*W*2)

        cls_score = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)
        return cls_score, bbox_pred

