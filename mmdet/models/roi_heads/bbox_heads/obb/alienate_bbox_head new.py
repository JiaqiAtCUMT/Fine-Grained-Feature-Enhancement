import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
import random
from mmdet.core import bbox
from mmdet.models.builder import HEADS
from .obbox_head import OBBoxHead
import torch.nn.functional as F

class RandomSuppress(nn.Module):
    def __init__(self,drop_threshold=0.25):
        self.drop_threshold=drop_threshold
        super().__init__()
    def forward(self, features):
        B,C,H,W = features.shape
        sum_features = features.sum(dim=1).reshape(B,H*W)
        drop_num = int(self.drop_threshold * H * W)
        index = random.sample(range(H * W),drop_num)
        mask = torch.ones_like(sum_features,device=features.device)
        mask.scatter_(dim=1,index=index,value=0.0)
        return features * mask.reshape(B,1,H,W)
class PeakSuppress(nn.Module):
    def __init__(self,drop_threshold=0.25):
        self.drop_threshold=drop_threshold
        super().__init__()
    def forward(self, features):
        B,C,H,W = features.shape
        sum_features = features.sum(dim=1).reshape(B,H*W)
        drop_num = int(self.drop_threshold * H * W)
        _,index = sum_features.topk(drop_num,dim=1,sorted=False)
        mask = torch.ones_like(sum_features,device=features.device)
        mask.scatter_(dim=1,index=index,value=0.0)
        return features * mask.reshape(B,1,H,W)
class AlienateModule(nn.Module):
    def __init__(self, in_channels, parallel_num = 4, r = 16):
        super().__init__()
        self.parallel_num=parallel_num
        self.parallel_convs = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels*self.parallel_num,
                3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                ),
        )
        self.parallel_channels = nn.Sequential(
            nn.Linear(in_channels,in_channels//r),
            nn.ReLU(),
            nn.Linear(in_channels//r,in_channels),
            nn.Sigmoid(),
        )
        self.combine_conv = ConvModule(
            in_channels*self.parallel_num,
            in_channels,
            1,
            padding=0,
            norm_cfg=dict(type='BN', requires_grad=True),
        )
    
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.loss_similarity = CosineSimilarity()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                xavier_init(m, distribution='uniform')
                
    def forward(self, features):
        B,C,H,W = features.shape
        x = self.parallel_convs(features)

        x = x.reshape(B,self.parallel_num,C,H,W)
        attn_maps = []
        attn_channels = []
        # scales = []
        for i in range(self.parallel_num):
            attn_map = x[:,i].sum(dim=1).sigmoid()[:,None,None,:,:] #x[:,i].mean(dim=1)[:,None,None,:,:]
            attn_channel = self.gap(x[:,i]).reshape(B,C)
            attn_channel = self.parallel_channels(attn_channel)
            # scales.append(attn_channel.mean(dim = 1)[:,None])
            attn_channels.append(attn_channel.reshape(B,1,C,1,1))
            attn_maps.append(attn_map)

        attn_maps = torch.cat(attn_maps,dim=1)
        attn_channels = torch.cat(attn_channels,dim=1)
        # scales = F.softmax(torch.cat(scales,dim=1),dim=1)[:,:,None,None,None]
        out = x * attn_maps * attn_channels
        out = self.combine_conv(out.reshape(B,-1,H,W))
        # out = (out * scales).sum(dim=1)

        return out 

@HEADS.register_module()
class AlienateBoxHead(OBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=1,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 parallel_num = 2,
                 peak_supress_cfg = dict(
                    weight=1.0,
                    drop_threshold=0.25,
                 ),
                 rand_supress_cfg = dict(
                    weight=1.0,
                    drop_threshold=0.25,
                 ),
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(AlienateBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.parallel_num = parallel_num
        self.supcon_loss = None
        self.peak_suppress = None
        if self.parallel_num>0:
            self.alienate_module = AlienateModule(self.in_channels,parallel_num)
        if peak_supress_cfg is not None:
            self.ps_weight = peak_supress_cfg['weight']
            self.peak_suppress = PeakSuppress(peak_supress_cfg['drop_threshold'])
        if rand_supress_cfg is not None:
            self.rs_weight = rand_supress_cfg['weight']
            self.rand_suppress = RandomSuppress(rand_supress_cfg['drop_threshold'])
        # self.multi_head = MultiheadAttention(self.in_channels,4)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels,is_cls=True)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = self.reg_dim if self.reg_class_agnostic else \
                    self.reg_dim * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False,is_cls=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            fc_out_channels = self.fc_out_channels if is_cls else self.fc_out_channels
            # print(fc_out_channels)
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else fc_out_channels

                branch_fcs.append(
                    nn.Linear(fc_in_channels, fc_out_channels))
            last_layer_dim = fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(AlienateBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    def cls_forward(self,x_cls):
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        return cls_score
    def reg_forward(self,x_reg):
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return bbox_pred
    def share_forward(self,x):
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x
    def forward(self, x):
        if self.parallel_num>0:
            x = self.alienate_module(x)
        ps_score = None
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        rs_feature = ps_feature = x
        x = self.share_forward(x)
        cls_feats = x
        # separate branches
        x_cls = x
        x_reg = x
        cls_score = self.cls_forward(x_cls)
        bbox_pred = self.reg_forward(x_reg)

        if self.training:
            if self.peak_suppress is not None:
                ps_feature = self.peak_suppress(ps_feature)
                ps_x = self.share_forward(ps_feature)
                ps_score = self.cls_forward(ps_x)
            if self.rand_suppress is not None:
                rs_feature = self.peak_suppress(rs_feature)
                rs_x = self.share_forward(rs_feature)
                rs_score = self.cls_forward(rs_x)

        return dict(cls_score=cls_score, bbox_pred=bbox_pred, 
            cls_feats=cls_feats ,ps_score=ps_score, rs_score = rs_score)
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, cls_feats, ps_score, rs_score, ious, reduction_override=None):
        loss =  super().loss(cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=reduction_override)
        if self.peak_suppress is not None:
            loss['loss_peak_suppress'] = self.ps_weight * self.loss_cls(
                        ps_score,
                        labels,
                        label_weights,
                        reduction_override=reduction_override)
        if self.rand_suppress is not None:
            loss['loss_rand_suppress'] = self.rs_weight * self.loss_cls(
                        rs_score,
                        labels,
                        label_weights,
                        reduction_override=reduction_override)
        if self.supcon_loss is not None:
            loss['loss_supcon'] = self.supcon_loss(cls_feats,labels,ious)  
        return loss

        