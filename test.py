import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.nn.modules.activation import MultiheadAttention

import cv2
import numpy as np

class ConLoss(nn.Module):
    """Contrastive LOSS"""

    def __init__(self, weight=0.1):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.weight = weight

    def forward(self, features):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        features = F.normalize(features,dim=1)
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label

        similarity = torch.matmul(features, features.T)
        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        # exp_sim = torch.exp(similarity) * logits_mask
        loss = similarity * logits_mask
        # loss = - torch.log(exp_sim.sum(dim=1, keepdim=True))
        return 1.0 - loss.sum() / (features.shape[0]-1) / features.shape[0]

h = 8
channels = 256
r = 16
height = weight = 14
batch_size = 4
seq_length = 10
groups = 8
input = torch.randn([batch_size, channels, height, weight])

conloss = ConLoss()
gap = nn.AdaptiveAvgPool2d(1)
conv1 = nn.Conv2d(in_channels=channels, out_channels=channels*groups, kernel_size=1, groups=1)
conv2 = nn.Conv2d(in_channels=channels*groups, out_channels=channels, kernel_size=1, groups=groups)
fc_channels = nn.Sequential(
    nn.Linear(channels,channels//r),
    nn.ReLU(),
    nn.Linear(channels//r,channels),
    nn.Sigmoid(),
)
x = conv1(input).reshape(batch_size,groups,channels,height,weight)
group_div = x.reshape(batch_size, groups, channels, height, weight)
attn_maps = []
attn_channels = []
scales = []
for i in range(groups):
    attn_map = group_div[:,i].sum(dim=1).sigmoid()[:,None,None,:,:]
    attn_maps.append(attn_map)
    attn_channel = gap(group_div[:,i]).reshape(batch_size,channels)
    attn_channel = fc_channels(attn_channel)
    scales.append(attn_channel.mean(dim = 1)[:,None])
    attn_channels.append(attn_channel.reshape(batch_size,1,channels,1,1))
attn_maps = torch.cat(attn_maps,dim=1)
loss = conloss(attn_maps.permute(1,0,2,3,4).reshape(groups,-1))
attn_channels = torch.cat(attn_channels,dim=1)
scales = F.softmax(torch.cat(scales,dim=1),dim=1)[:,:,None,None,None]
out = x * attn_maps * attn_channels
out = (out * scales).sum(dim=1)
print(out.shape)
# print(loss)
# print(x.shape)
# x = conv2(x)
# print(x.shape)
# def partition(x, size):
#     """
#     Args:
#         x: (B, C, H, W)
#         size (int): partition size

#     Returns:
#         partition: (B,C,h_num_partition,w_num_partition, size, size)
#     """
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // size, size, W // size, size)
#     return x.permute(0, 1, 2, 4, 3, 5).contiguous()


# topnum = 4
# feats = torch.rand((3,5,3,3))
# B,C,H,W = feats.shape
# sum_feats = feats.sum(dim=1)
# _,ids = sum_feats.reshape(B,H*W).topk(topnum)
# print(ids)
# input = torch.rand((3,5,9,9))

# p = partition(input,3).reshape(B,C,9,3,3)
# print(p.shape)
# t = []
# for i in range(B):
#     t.append(p[None,i,:,ids[i]])
#     print(t[i].shape)
# print(torch.cat(t,dim=0).shape)

# pix_attn_conv = nn.Conv2d(channels,1,3,padding=1)
# gap = nn.AdaptiveAvgPool2d(1)
# input = torch.randn([batch_size, channels, height, weight])
# target = torch.randn([batch_size, 1, height, weight])
# channel_attn = gap(input)
# pix_attn = pix_attn_conv(input)
# out = pix_attn * pix_attn * input
# loss = nn.BCEWithLogitsLoss()

# print(out.shape)
# print(pix_attn.shape,target.shape)
# print(loss(pix_attn,target))
# mask = np.zeros((1024,1024,1))
# area1 = np.array([[[250, 200], [300, 100], [750, 800], [100, 1000]]])
# print(area1.shape)
# mask = cv2.fillPoly(mask,area1,255)
# cv2.imwrite("a.png",mask)

# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim = -1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn

# class MultiHeadAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % h == 0
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(4)])
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#         batch_size = query.size(0)

#         query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
#         query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

#         x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
#         return self.linears[-1](x)



# model = MultiheadAttention(channels,h)
# input = torch.randn([batch_size, channels, height, weight])
# query = input.permute(2,3,0,1).reshape(height*weight,batch_size,channels)
# key = query
# value = query

# print ('Input size: ' + str(input.size()))

# m,w = model(query, key, value)
# print(m)
# m = m.permute(1,2,0).reshape(batch_size,channels,height,weight)
# print ('Output size: ' + str(m.size()))

# h = 8
# channels = 256
# height = weight = 7
# batch_size = 1
# seq_length = 10
# input = torch.randn([batch_size, channels, height, weight])
# gap = nn.AdaptiveAvgPool2d(3)
# gap1 = nn.AdaptiveAvgPool1d(1)
# input = input.reshape(batch_size, channels, height*weight).permute(0,2,1)
# cls_branch = gap1(input).reshape(batch_size,height,weight)
# print(cls_branch.shape)