import re
import os
import time
import zipfile

import os.path as osp
import numpy as np

from PIL import Image
from functools import reduce, partial
from multiprocessing import Pool
from collections import defaultdict

from .io import load_imgs
from .misc import get_classes, img_exts
from ..utils import get_bbox_type
from ..geometry import bbox2type
from tqdm import tqdm

def load_fair1m(img_dir, ann_dir=None, classes=None, nproc=10):
    classes = get_classes('FAIR1M' if classes is None else classes)
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    print('Starting loading FAIR1M dataset information.')
    start_time = time.time()
    _load_func = partial(_load_fair1m_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir,
                         cls2lbl=cls2lbl)

    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    contents = [c for c in contents if c is not None]
    end_time = time.time()
    print(f'Finishing loading FAIR1M, get {len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')

    return contents, classes


def _load_fair1m_single(imgfile, img_dir, ann_dir, cls2lbl):
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    txtfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.txt')
    content = _load_fair1m_txt(txtfile, cls2lbl)

    if ann_dir is not None and content is None:
        return None

    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size

    content.update(dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def _load_fair1m_txt(txtfile, cls2lbl):
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.exists(txtfile):
        return None
        # print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                items = line.split(' ')
                if len(items) >= 9:
                    if items[8] not in cls2lbl:
                        assert f'{items[8]} not in cls2lbl'
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(cls2lbl[items[8]])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
            np.zeros((0, 8), dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) if labels else \
            np.zeros((0, ), dtype=np.int64)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
            np.zeros((0, ), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)


def load_fair1m_submission(ann_dir, img_dir=None, classes=None, nproc=10):
    classes = get_classes('FAIR1M' if classes is None else classes)

    file_pattern = r'Task[1|2]_(.*)\.txt'
    cls2file_mapper = dict()
    for f in os.listdir(ann_dir):
        match_objs = re.match(file_pattern, f)
        if match_objs is None:
            fname, _ = osp.splitext(f)
            cls2file_mapper[fname] = f
        else:
            cls2file_mapper[match_objs.group(1)] = f

    print('Starting loading FAIR1M submission information')
    start_time = time.time()
    infos_per_cls = []
    for cls in classes:
        if cls not in cls2file_mapper:
            infos_per_cls.append(dict())
        else:
            subfile = osp.join(ann_dir, cls2file_mapper[cls])
            infos_per_cls.append(_load_fair1m_submission_txt(subfile))

    if img_dir is not None:
        contents, _ = load_imgs(img_dir, nproc=nproc, def_bbox_type='poly')
    else:
        all_id = reduce(lambda x, y: x|y, [d.keys() for d in infos_per_cls])
        contents = [{'id':i} for i in all_id]

    for content in contents:
        bboxes, scores, labels = [], [], []
        for i, infos_dict in enumerate(infos_per_cls):
            infos = infos_dict.get(content['id'], dict())
            bboxes.append(infos.get('bboxes', np.zeros((0, 8), dtype=np.float32)))
            scores.append(infos.get('scores', np.zeros((0, ), dtype=np.float32)))
            labels.append(np.zeros((bboxes[-1].shape[0], ), dtype=np.int64) + i)

        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)
        content['ann'] = dict(bboxes=bboxes, labels=labels, scores=scores)
    end_time = time.time()
    print(f'Finishing loading FAIR1M submission, get{len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')
    return contents, classes


def _load_fair1m_submission_txt(subfile):
    if not osp.exists(subfile):
        print(f"Can't find {subfile}, treated as empty subfile")
        return dict()

    collector = defaultdict(list)
    with open(subfile, 'r') as f:
        for line in f:
            img_id, score, *bboxes = line.split(' ')
            bboxes_info = bboxes + [score]
            bboxes_info = [float(i) for i in bboxes_info]
            collector[img_id].append(bboxes_info)

    anns_dict = dict()
    for img_id, info_list in collector.items():
        infos = np.array(info_list, dtype=np.float32)
        bboxes, scores = infos[:, :-1], infos[:, -1]
        bboxes = bbox2type(bboxes, 'poly')
        anns_dict[img_id] = dict(bboxes=bboxes, scores=scores)
    return anns_dict


def save_fair1m_submission(save_dir, id_list, dets_list, task='Task1', classes=None, with_zipfile=True):
    assert task in ['Task1', 'Task2']
    classes = get_classes('FAIR1M' if classes is None else classes)

    # if osp.exists(save_dir):
    #     raise ValueError(f'The save_dir should be a non-exist path, but {save_dir} is existing')
    os.makedirs(save_dir,exist_ok=True)
    

    for img_id, dets_per_cls in tqdm(zip(id_list, dets_list)):
        pred_list = []
        for i, dets in enumerate(dets_per_cls):
            bboxes, scores = dets[:, :-1], dets[:, -1]

            if task == 'Task1':
                if get_bbox_type(bboxes) == 'poly' and bboxes.shape[-1] != 8:
                    bboxes = bbox2type(bboxes, 'obb')
                bboxes = bbox2type(bboxes, 'poly')
            else:
                bboxes = bbox2type(bboxes, 'hbb')

            for bbox, score in zip(bboxes, scores):
                pred_list.append({"category":classes[i],"points":bbox,"prob":score})
        save_xml(img_id,pred_list,save_dir)


def save_xml(img_id,pred,out_folder):
    str = ""
    str+='<?xml version="1.0" encoding="utf-8"?>\n'
    str+='<annotation>\n'
    str+='    <source>\n'
    str+=f'       <filename>{img_id}.tif</filename>\n'
    str+='       <origin>GF2/GF3</origin>\n'
    str+='    </source>\n'
    str+='    <research>\n'
    str+='        <version>1.0</version>\n'
    str+='        <provider>CUMT</provider>\n'
    str+='        <author>cumt_aicv_yyds</author>\n'
    str+='        <pluginname>FAIR1M</pluginname>\n'
    str+='        <pluginclass>object detection</pluginclass>\n'
    str+='        <time>2021-08</time>\n'
    str+='    </research>\n'
    str+='    <objects>\n'
    for v in pred:
        v["category"] = v["category"].replace('_',' ')
        str+='        <object>\n'
        str+='            <coordinate>pixel</coordinate>\n'
        str+='            <type>rectangle</type>\n'
        str+='            <description>None</description>\n'
        str+='            <possibleresult>\n'
        str+=f'                <name>{v["category"]}</name>                \n'
        str+=f'                <probability>{v["prob"]}</probability>\n'
        str+='            </possibleresult>\n'
        str+='            <points>  \n'
        str+=f'                <point>{v["points"][0]}, {v["points"][1]}</point>\n'
        str+=f'                <point>{v["points"][2]}, {v["points"][3]}</point>\n'
        str+=f'                <point>{v["points"][4]}, {v["points"][5]}</point>\n'
        str+=f'                <point>{v["points"][6]}, {v["points"][7]}</point>\n'
        str+=f'                <point>{v["points"][0]}, {v["points"][1]}</point>\n'
        str+='            </points>\n'
        str+='        </object>\n'
    str +='     </objects>\n'
    str +=' </annotation>\n'
    fname = f'{img_id}.xml'
    fname = osp.join(out_folder, fname)
    with open(fname, "w") as f:
        f.write(str)