import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginModel(nn.Module):
    def __init__(self, m=0.5,s=64,num_classes= 37,emb_size=1024,easy_margin=False):
        super(ArcMarginModel, self).__init__()
 
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        # emb_size 特征向量长度
        nn.init.xavier_uniform_(self.weight)
        # 使用均匀分布来初始化weight
        self.easy_margin = easy_margin
        self.m = m
        # 夹角差值 0.5 公式中的m
        self.s = s
        # 半径 64 公式中的s
        # 二者大小都是论文中推荐值
 
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # 差值的cos和sin
        self.th = math.cos(math.pi - self.m)
        # 阈值，避免theta + m >= pi
        self.mm = math.sin(math.pi - self.m) * self.m
 
    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        # 正则化
        cosine = F.linear(x, W)
        # cos值
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # sin
        phi = cosine * self.cos_m - sine * self.sin_m
        # cos(theta + m) 余弦公式
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            # 如果使用easy_margin
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        output *= self.s
        # 乘以半径
        return output

