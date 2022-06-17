import os,shutil
import os.path as osp
from glob import glob
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
map_dict = {
    'A321':'A320/321',
    'Boeing737':'Boeing737-800',
    'Boeing787':'Boeing787',
    'A220':'A220',
    'ARJ21':'ARJ21',
    'A330':'A330',
    'other-airplane':'other'
}

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get(root, name):
    return root.findall(name)
def clip(val,min,max):
    if val<min:
        return min
    if val>max:
        return max
    return val
def filter_airplane(srcpath,dstpath):
    xml_filelist = glob(osp.join(srcpath,'labelXml','*.xml'))
    tar_txt_path = osp.join(dstpath,'labelTxt')
    tar_img_path = osp.join(dstpath,'images')
    src_img_path = osp.join(srcpath,"images")
    count = 0
    os.makedirs(tar_txt_path,exist_ok=True)
    os.makedirs(tar_img_path,exist_ok=True)

    for file in tqdm(xml_filelist):
        tree = ET.parse(file)
        root = tree.getroot()
        objects = get_and_check(root,'objects',1)
        size = get_and_check(root,'size',1)
        height = float(get_and_check(size, 'height',1).text)
        width = float(get_and_check(size, 'width',1).text)
        gts = []
        flag = False
        for obj in get(objects,'object'):
            possibleresult = get_and_check(obj, 'possibleresult', 1)
            category = get_and_check(possibleresult, 'name', 1).text
            if category not in map_dict:
                continue
            flag = True
            points = get_and_check(obj, 'points', 1)
            points = get(points, 'point')
            points = [pt.text.split(',') for pt in points][:4]
            gt = []
            for pt in points:
                pt[0] = clip(float(pt[0]),0.,width)
                pt[1] = clip(float(pt[1]),0.,height)
                gt.extend(pt)
            gt.append(map_dict[category])
            gt.append(0)
            gts.append(' '.join(map(str,gt))+'\n')
        if flag:
            count+=1
            base,_ = osp.splitext(osp.basename(file))
            img_src = osp.join(src_img_path,base+'.tif')
            base += '_fair1m'
            with open(os.path.join(tar_txt_path,base+'.txt') ,'w') as save_f:
                save_f.writelines(gts)
            img = cv2.imread(img_src)
            cv2.imwrite(osp.join(tar_img_path,base+'.png'),img)
    print('Convert number:',count)
# filter_airplane(r'/root/datasets/FAIR1M/train/',r'/root/datasets/FAIR1M-AIR')

src_path = r'/root/datasets/SARAR_AUG_ALL'
img_path = src_path+'/images'
lbl_path = src_path+'/labelTxt'
dst_img_path = r'/root/datasets/sarar_refine/images'
dst_lbl_path = r'/root/datasets/sarar_refine/labelTxt'
os.makedirs(dst_img_path,exist_ok=True)
os.makedirs(dst_lbl_path,exist_ok=True)
with open('output_att.txt','r') as f:
    lines = f.readlines()
for line in lines:
    img_id ,mAP =line.split(' ')[:2]
    img_id = img_id.split(':')[-1]
    mAP = float(mAP.split(':')[-1])
    if mAP>=0.2 and mAP<=0.7:
        print(img_id)
        shutil.copy(osp.join(img_path,img_id+'.png'),osp.join(dst_img_path,img_id+'.png'))
        shutil.copy(osp.join(lbl_path,img_id+'.txt'),osp.join(dst_lbl_path,img_id+'.txt'))




# src_path = r'/root/datasets/SARAR-MIX'
# txt_filelist = glob(osp.join(src_path,'labelTxt','*.txt'))
# for file in txt_filelist:
#     base,_ = osp.splitext(osp.basename(file))
#     if '-att' in base:
#         print(base)
#         os.remove(file)
#         os.remove(osp.join(src_path,'images',base+'.png'))