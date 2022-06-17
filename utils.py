import numpy as np
import torch
import cv2
from torch.functional import Tensor

pi = 3.141592


def get_bbox_type(bboxes, with_score=False):
    dim = bboxes.size(-1)
    if with_score:
        dim -= 1

    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim  == 8:
        return 'poly'
    return 'notype'


def get_bbox_dim(bbox_type, with_score=False):
    if bbox_type == 'hbb':
        dim = 4
    elif bbox_type == 'obb':
        dim = 5
    elif bbox_type == 'poly':
        dim = 8
    else:
        raise ValueError(f"don't know {bbox_type} bbox dim")

    if with_score:
        dim += 1
    return dim


def get_bbox_areas(bboxes):
    btype = get_bbox_type(bboxes)
    if btype == 'hbb':
        wh = bboxes[..., 2:] - bboxes[..., :2]
        areas = wh[..., 0] * wh[..., 1]
    elif btype == 'obb':
        areas = bboxes[..., 2] * bboxes[..., 3]
    elif btype == 'poly':
        pts = bboxes.view(*bboxes.size()[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(pts[..., 0] * roll_pts[..., 1] -
                         roll_pts[..., 0] * pts[..., 1], dim=-1)
        areas = 0.5 * torch.abs(xyxy)
    else:
        raise ValueError('The type of bboxes is notype')

    return areas


def choice_by_type(hbb_op, obb_op, poly_op, bboxes_or_type,
                   with_score=False):
    if isinstance(bboxes_or_type, torch.Tensor):
        bbox_type = get_bbox_type(bboxes_or_type, with_score)
    elif isinstance(bboxes_or_type, str):
        bbox_type = bboxes_or_type
    else:
        raise TypeError(f'need np.ndarray or str,',
                        f'but get {type(bboxes_or_type)}')

    if bbox_type == 'hbb':
        return hbb_op
    elif bbox_type == 'obb':
        return obb_op
    elif bbox_type == 'poly':
        return poly_op
    else:
        raise ValueError('notype bboxes is not suppert')


def arb2result(bboxes, labels, num_classes, bbox_type='hbb'):
    assert bbox_type in ['hbb', 'obb', 'poly']
    bbox_dim = get_bbox_dim(bbox_type, with_score=True)

    if bboxes.shape[0] == 0:
        return [np.zeros((0, bbox_dim), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def arb2roi(bbox_list, bbox_type='hbb'):
    assert bbox_type in ['hbb', 'obb', 'poly']
    bbox_dim = get_bbox_dim(bbox_type)

    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :bbox_dim]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, bbox_dim+1))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def distance2obb(points, distance, max_shape=None):
    distance, theta = distance.split([4, 1], dim=1)

    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.cat([Cos, Sin, -Sin, Cos], dim=1).reshape(-1, 2, 2)

    wh = distance[:, :2] + distance[:, 2:]
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset_t = offset_t.unsqueeze(2)
    offset = torch.bmm(Matrix, offset_t).squeeze(2)
    ctr = points + offset

    obbs = torch.cat([ctr, wh, theta], dim=1)
    return regular_obb(obbs)


def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def regular_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    w_regular = torch.where(w > h, w, h)
    h_regular = torch.where(w > h, h, w)
    theta_regular = torch.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


def mintheta_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + pi/2)
    abs_theta1 = torch.abs(theta1)
    abs_theta2 = torch.abs(theta2)

    w_regular = torch.where(abs_theta1 < abs_theta2, w, h)
    h_regular = torch.where(abs_theta1 < abs_theta2, h, w)
    theta_regular = torch.where(abs_theta1 < abs_theta2, theta1, theta2)

    obboxes = torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
    return obboxes

def poly2obb(polys):
    if isinstance(polys,Tensor):
        polys_np = polys.detach().cpu().to().numpy()
    elif isinstance(polys,np.ndarray):
        polys_np = polys
    else:
        polys_np = np.array(polys)
    order = polys_np.shape[:-1]
    num_points = polys_np.shape[-1] // 2
    polys_np = polys_np.reshape(-1, num_points, 2)
    polys_np = polys_np.astype(np.float32)

    obboxes = []
    for poly in polys_np:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * pi
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)

    obboxes = obboxes.reshape(*order, 5)
    return torch.tensor(obboxes,dtype=torch.float)


def rectpoly2obb(polys):
    theta = torch.atan2(-(polys[..., 3] - polys[..., 1]),
                        polys[..., 2] - polys[..., 0])
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.stack([Cos, -Sin, Sin, Cos], dim=-1)
    Matrix = Matrix.view(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = torch.stack([x, y], dim=-1).unsqueeze(-2)
    center_polys = polys.view(*polys.shape[:-1], 4, 2) - center
    rotate_polys = torch.matmul(center_polys, Matrix.transpose(-1, -2))

    xmin, _ = torch.min(rotate_polys[..., :, 0], dim=-1)
    xmax, _ = torch.max(rotate_polys[..., :, 0], dim=-1)
    ymin, _ = torch.min(rotate_polys[..., :, 1], dim=-1)
    ymax, _ = torch.max(rotate_polys[..., :, 1], dim=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = torch.stack([x, y, w, h, theta], dim=-1)
    return regular_obb(obboxes)


def poly2hbb(polys):
    polys = polys.view(*polys.shape[:-1], polys.size(-1)//2, 2)
    lt_point = torch.min(polys, dim=-2)[0]
    rb_point = torch.max(polys, dim=-2)[0]
    return torch.cat([lt_point, rb_point], dim=-1)
    
def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat(
        [w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat(
        [-h/2 * Sin, -h/2 * Cos], dim=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat(
        [point1, point2, point3, point4], dim=-1)


def obb2hbb(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w/2 * Cos) + torch.abs(h/2 * Sin)
    y_bias = torch.abs(w/2 * Sin) + torch.abs(h/2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center-bias, center+bias], dim=-1)


def hbb2poly(hbboxes):
    l, t, r, b = hbboxes.unbind(-1)
    return torch.stack([l, t, r, t, r, b, l ,b], dim=-1)


def hbb2obb(hbboxes):
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)

    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta-pi/2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


_type_func_map = {
    ('poly', 'obb'): poly2obb,
    ('poly', 'hbb'): poly2hbb,
    ('obb', 'poly'): obb2poly,
    ('obb', 'hbb'): obb2hbb,
    ('hbb', 'poly'): hbb2poly,
    ('hbb', 'obb'): hbb2obb
}


def bbox2type(bboxes, to_type):
    assert to_type in ['hbb', 'obb', 'poly']

    ori_type = get_bbox_type(bboxes)
    if ori_type == 'notype':
        raise ValueError('Not a bbox type')
    if ori_type == to_type:
        return bboxes
    trans_func = _type_func_map[(ori_type, to_type)]
    return trans_func(bboxes)

def poly2points(poly):
    return poly.reshape((-1,2)).numpy().astype(int)
def points2poly(points):
    return torch.tensor(points,dtype=torch.float32).reshape(-1)

def expend_obb(obbox,padding):
    if isinstance(padding,float):
        obbox[2] *= (1+padding)
        obbox[3] *= (1+padding)
    elif isinstance(padding,int):
        obbox[2] += padding
        obbox[3] += padding
    return obbox
def expend_poly(poly,padding):
    return obb2poly(expend_obb(poly2obb(poly),padding))