import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropyLoss
from ..builder import LOSSES
class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        arc_cos = torch.acos(cosine)
        M = F.one_hot(label, num_classes = cosine.shape[1] ) * self.m
        arc_cos = arc_cos + M
        
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.s
        return logits
@LOSSES.register_module()
class ArcFaceLoss(CrossEntropyLoss):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 scale=64,
                 margin=0.5):
        """ArcFaceLoss

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(ArcFaceLoss, self).__init__(use_sigmoid,use_mask,reduction,class_weight,loss_weight)
        self.arcface = ArcFace(scale,margin)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        cls_score = self.arcface(cls_score,label)
        return super(ArcFaceLoss,self).forward(cls_score,label,weight,avg_factor,reduction_override,**kwargs)
