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

class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func=None, weight=0.1):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.weight = weight

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]
        features = F.normalize(features,dim=1)
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return self.weight * loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option is None:
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

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
    def __init__(self, in_channels, parallel_num = 2):
        super().__init__()
        self.parallel_num=parallel_num
        self.parallel_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                in_channels*self.parallel_num,
                3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                ),
        ])

        self.combine_conv = ConvModule(
                in_channels*self.parallel_num,
                in_channels,
                1,
                padding=0,
                norm_cfg=dict(type='BN', requires_grad=True),
                groups=self.parallel_num
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
                
    def forward(self, features):
        # B,C,H,W = features.shape
        x = features
        for conv in self.parallel_convs:
            x = conv(x)
        x = self.combine_conv(x)
        return x

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
                 supress_cfg = dict(
                    suppress_weight=0.2,
                    drop_threshold=0.7,
                 ),
                 supcon_cfg =dict(
                    weight=0.05,
                    temperature=1.0,
                    iou_threshold=0.7,
                    reweight_func='linear',
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
        self.alienate_module = AlienateModule(self.in_channels,parallel_num)
        if supress_cfg is not None:
            self.suppress_weight = supress_cfg['suppress_weight']
            self.peak_suppress = PeakSuppress(supress_cfg['drop_threshold'])
        if supcon_cfg is not None:
            self.supcon_loss = SupConLoss(**supcon_cfg)
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
        x = self.alienate_module(x)
        ps_score = None
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        # query = key = value = x.permute(2,3,0,1).reshape(self.roi_feat_area, -1, self.in_channels)
        # attn_out,_ = self.multi_head(query,key,value)
        # x = attn_out.permute(1,2,0).reshape(-1,self.in_channels,self.roi_feat_size[0],self.roi_feat_size[1])
        
        ps_feature = x
        x = self.share_forward(x)
        cls_feats = x
        # separate branches
        x_cls = x
        x_reg = x
        cls_score = self.cls_forward(x_cls)
        bbox_pred = self.reg_forward(x_reg)

        if self.training and self.peak_suppress is not None:
            ps_feature = self.peak_suppress(ps_feature)
            ps_x = self.share_forward(ps_feature)
            ps_score = self.cls_forward(ps_x)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, 
            cls_feats=cls_feats ,ps_score=ps_score)
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, cls_feats, ps_score, ious, reduction_override=None):
        loss =  super().loss(cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=reduction_override)
        if self.peak_suppress is not None:
            loss['loss_suppress'] = self.suppress_weight * self.loss_cls(
                        ps_score,
                        labels,
                        label_weights,
                        reduction_override=reduction_override)
        if self.supcon_loss is not None:
            loss['loss_supcon'] = self.supcon_loss(cls_feats,labels,ious)  
        return loss

        