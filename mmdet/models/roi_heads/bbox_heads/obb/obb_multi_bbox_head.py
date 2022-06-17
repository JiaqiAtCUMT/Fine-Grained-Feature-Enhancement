import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .obbox_head import OBBoxHead
from .obb_convfc_bbox_head import OBBShared2FCBBoxHead
from mmdet.core import force_fp32,get_bbox_dim,multiclass_arb_nms
from mmdet.models.losses import accuracy
import torch.nn.functional as F

def correct_num(pred, target, topk=1):
    """Calculate accuracy according to the prediction and target

    Args:
        pred (torch.Tensor): The model prediction.
        target (torch.Tensor): The target of each prediction
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res[0] if return_single else res


class OBBSubcategoryBBoxHead(OBBShared2FCBBoxHead):
    
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605
    def __init__(self,level,
                 *args,
                 **kwargs):
        self.level = level
        super(OBBShared2FCBBoxHead, self).__init__(*args, **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        neg_inds = (labels < 0) | (labels >= bg_class_ind)
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)            
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            losses['pos_num'] = pos_inds.sum().float()
            losses['neg_num'] = (pos_inds.numel() - losses['pos_num']).float()
            losses['pos_det_num'] = correct_num(cls_score[pos_inds], labels[pos_inds]) if pos_inds.any() else 0
            losses['neg_det_num'] = correct_num(cls_score[neg_inds], labels[neg_inds]) if neg_inds.any() else 0
            # do not perform bounding box regression for BG anymore.
            target_dim = self.reg_dim
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    target_dim = get_bbox_dim(self.end_bbox_type)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), target_dim)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        target_dim)[pos_inds.type(torch.bool),
                                    labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses

@HEADS.register_module()
class OBBMultiBBoxHead(OBBoxHead):

    def get_subcategories_info(self,categories):
        idx2subcategory = {}
        len_subcategories = []
        subcategories2idx = []
        idx = 0
        for i,(k,sub) in enumerate(categories):
            subcategories2idx.append([])
            for j in range(len(sub)):
                idx2subcategory[idx] = i,j
                subcategories2idx[i].append(idx)
                idx += 1
            len_subcategories.append(len(sub))
        return idx2subcategory, len_subcategories, subcategories2idx, idx
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605
    def __init__(self,
                 num_shared_convs=0,
                 num_sub_shared_convs=0,
                 num_sub_shared_fcs=0,
                 num_sub_cls_convs=0,
                 num_sub_cls_fcs=0,
                 num_sub_reg_convs=0,
                 num_sub_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 categories = None,
                 *args,
                 **kwargs):
        super(OBBMultiBBoxHead, self).__init__(*args, **kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_sub_shared_convs = num_sub_shared_convs
        self.num_sub_shared_fcs = num_sub_shared_fcs
        self.num_sub_cls_convs = num_sub_cls_convs
        self.num_sub_cls_fcs = num_sub_cls_fcs
        self.num_sub_reg_convs = num_sub_reg_convs
        self.num_sub_reg_fcs = num_sub_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.categories = categories
        self.idx2subcategory, self.len_subcategories, self.subcategories2idx, self.num_classes = self.get_subcategories_info(categories)

        self.heads = nn.ModuleList()
        for i,length in enumerate(self.len_subcategories):
            self.heads.append(
                OBBSubcategoryBBoxHead(
                    level=i,num_classes = length,num_shared_convs = num_sub_shared_convs, num_shared_fcs=num_sub_shared_fcs,
                    num_cls_convs = num_sub_cls_convs, num_cls_fcs = num_sub_cls_fcs,
                    num_reg_convs = num_sub_reg_convs, num_reg_fcs= num_sub_reg_fcs,
                    conv_out_channels = conv_out_channels, fc_out_channels = fc_out_channels,
                    conv_cfg = conv_cfg,norm_cfg = norm_cfg,*args,**kwargs)
            )
    def init_weights(self):
        for head in self.heads:
            head.init_weights()

    def map_subcategory_index(self,head_id,label):
        if self.num_classes == label:
            return self.len_subcategories[head_id]
        depart_level, sub_index = self.idx2subcategory[label]
        if head_id!=depart_level:
            return self.len_subcategories[head_id]
        return sub_index

    def forward(self, x, pred_inds=None):
        cls_results = []
        reg_results = []
        for i,head in enumerate(self.heads):
            cls_result,reg_result = head(x)
            if not self.training:                
                offset = self.subcategories2idx[i][0]
                new_cls_result = torch.zeros((cls_result.shape[0],self.num_classes + 1),device=cls_result.device)
                new_cls_result[:,offset:offset+cls_result.shape[1]-1] = cls_result[:,:-1]
                new_cls_result[:,-1] = cls_result[:,-1]
                cls_result = new_cls_result

            cls_results.append(cls_result)
            reg_results.append(reg_result)
        
        return cls_results,reg_results
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        multi_loss = None
        for i,(cls,bbox,head) in enumerate(zip(cls_score,bbox_pred,self.heads)):
            
            gt_label_cpu = labels.cpu()
            gt_label_cpu = gt_label_cpu.map_(gt_label_cpu,lambda x,y:self.map_subcategory_index(i,x))
            new_gt_labels= gt_label_cpu.to(labels.device)
            head_loss = head.loss(cls, bbox, rois, new_gt_labels, label_weights, bbox_targets, bbox_weights, reduction_override=reduction_override)
            if multi_loss is None:
                multi_loss = head_loss
            else:
                for k in head_loss.keys():
                    multi_loss[k] += head_loss[k]
        multi_loss['loss_cls'] /= len(self.len_subcategories)
        multi_loss['loss_bbox'] /= len(self.len_subcategories)
        multi_loss['acc'] /= len(self.len_subcategories)
        multi_loss['pos_acc'] = multi_loss['pos_det_num']/multi_loss['pos_num'] * 100.0
        multi_loss['neg_acc'] = multi_loss['neg_det_num']/multi_loss['neg_num'] * 100.0
        
        return multi_loss
