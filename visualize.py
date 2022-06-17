
from cProfile import label
import glob
import os,cv2
from PIL.Image import ORDERED
import xml.etree.ElementTree as ET
from glob import glob
import numpy as np
import random
# CATEGORY = {
#     'AirPlane':[
#         'A220', 'A321', 'A330', 'A350', 'ARJ21',
#         'Boeing737','Boeing747', 'Boeing777', 'Boeing787','C919',
#         'other-airplane'
#     ],
#     'Court':[
#         'Baseball Field', 'Basketball Court','Football Field','Tennis Court'
#     ],
#     'Road':[
#         'Bridge','Roundabout','Intersection',
#     ],
#     'Ship':[
#         'Dry Cargo Ship','Passenger Ship',
#         'Motorboat','Fishing Boat', 'Tugboat', 'Engineering Ship',
#         'Liquid Cargo Ship', 'Warship',
#         'other-ship'
#     ],
#     'Vehicle':[
#         'Bus','Dump Truck', 'Excavator', 'Small Car', 
#         'Tractor','Trailer','Truck Tractor', 'Van',
#         'Cargo Truck','other-vehicle'
#     ],
# }


# CATEGORY_FIND = {}
# for cls,val in CATEGORY.items():
#     for subclass in val:
#         CATEGORY_FIND[subclass] = cls
# COLOR_MAP = {
#     'AirPlane':(0,255,255),#yellow
#     'Ship':(255,0,0),#blue
#     'Vehicle':(255,255,0),#light blue
#     'Court':(0,0,255),#red
#     'Road':(0,255,0),#green
# }
COLOR_MAP = {
    # 'C919':(0,255,255),#yellow
    'Baseball Field':(255,255,255),#white
    'Football Field':(255,255,0),#light blue
    'Tennis Court':(0,255,0),#green
    'default':(0,0,0)
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

def getBaseName(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def draw_rect(file,img,threshold = 0.1,random_drop=0):
    tree = ET.parse(file)
    root = tree.getroot()
    objects = get_and_check(root,'objects',1)
    id = 0
    for obj in get(objects,'object'):
        id+=1
        possibleresult = get_and_check(obj, 'possibleresult', 1)
        category = get_and_check(possibleresult, 'name', 1).text
        probability = float(get_and_check(possibleresult, 'probability', 1).text)
        if(probability<threshold):
            continue
        if category =='Baseball Field' and random.random()<random_drop:
            continue
        points = get(get_and_check(obj, 'points', 1), 'point')
        points = [list(map(float,pt.text.split(","))) for pt in points]
        pts = np.array(points,dtype=np.int32)[:4]
        # color = COLOR_MAP[CATEGORY_FIND[category]]
        color = COLOR_MAP[category] if category in COLOR_MAP else COLOR_MAP['default']
        cv2.polylines(img, [pts], 1, color,2)
    return img
def obj_img_extract(labelpath,imgspath,dst_path,gt_label_path=None,order_set = None,threshold = 0.1,random_drop=0):
    filelist = glob(os.path.join(labelpath,'*.xml')) 
    os.makedirs(dst_path,exist_ok=True)
    for file in filelist:
        base_name = getBaseName(file)
        if order_set!=None and base_name not in order_set:
            continue
        print(base_name)
        img = cv2.imread(os.path.join(imgspath,base_name +'.tif'))
        # cv2.imwrite(os.path.join(dst_path,f"preview_{base_name}_gt.png"),draw_rect(os.path.join(gt_label_path,base_name+'.xml'),img.copy()))
        img = draw_rect(file,img, threshold,random_drop)
        cv2.imwrite(os.path.join(dst_path,f"preview_{base_name}.png"),img)

def obj_select(labelpath,target_class,threshold=0.5):
    filelist = glob(os.path.join(labelpath,'*.xml')) 
    result = []
    for file in filelist:
        base_name = getBaseName(file)
        tree = ET.parse(file)
        root = tree.getroot()
        objects = get_and_check(root,'objects',1)
        for obj in get(objects,'object'):
            possibleresult = get_and_check(obj, 'possibleresult', 1)
            category = get_and_check(possibleresult, 'name', 1).text
            probability = float(get_and_check(possibleresult, 'probability', 1).text)
            if(category==target_class and probability>=threshold):
                result.append(base_name)
                break
    return result

def obj_select_less(labelpath,target_class,threshold=0.1):
    filelist = glob(os.path.join(labelpath,'*.xml')) 
    result = []
    for file in filelist:
        base_name = getBaseName(file)
        tree = ET.parse(file)
        root = tree.getroot()
        objects = get_and_check(root,'objects',1)
        for obj in get(objects,'object'):
            possibleresult = get_and_check(obj, 'possibleresult', 1)
            category = get_and_check(possibleresult, 'name', 1).text
            probability = float(get_and_check(possibleresult, 'probability', 1).text)
            if(category==target_class and probability<threshold):
                result.append(base_name)
                break
    return result
# label_path = r"D:\datasets\FAIR1M\train\l"
# gt_label_path = r"/root/datasets/FAIR1M-SM/labelXmls/"
img_path = r"D:\datasets\FAIR1M\test\images"
# dst_path = r"visual_result"

# obj_img_extract(label_path,img_path,dst_path)
label_path_a = r"work_dirs\faster_rcnn_orpn_r50_fpn_1x_fair1m_ms_al_n1\test"
label_path_b = r"work_dirs\faster_rcnn_orpn_r50_fpn_1x_fair1m\test"
dst_path_a = r"work_dirs\faster_rcnn_orpn_r50_fpn_1x_fair1m_ms_al_n1\visual"
dst_path_b = r"work_dirs\faster_rcnn_orpn_r50_fpn_1x_fair1m\visual"
# obj_img_extract(label_path,r"D:\datasets\FAIR1M\test\images",dst_path)
target_class = "Baseball Field"
# a = set(obj_select(label_path_a,target_class))
# b = set(obj_select_less(label_path_b,target_class))
# dif = a & b
dif = set(["1737"])
obj_img_extract(label_path_a,img_path,dst_path_a,order_set= dif)
obj_img_extract(label_path_b,img_path,dst_path_b,order_set= dif,random_drop=0)
