from collections import defaultdict
from posixpath import basename
import cv2
import os
import shutil
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import img_split
def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles


def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = os.path.basename(os.path.splitext(file)[0])
            f_out.write(basename + '\n')

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

def clip(num,low,upper):
    if num<low:
        return low
    if num>upper:
        return upper
    return num

def xml2txt(srcpath,dstpath):
    filelist = GetFileFromThisRootDir(srcpath)
    if not os.path.isdir(dstpath):
        os.mkdir(dstpath)
    for file in filelist:
        with open(os.path.join(dstpath,os.path.split(file)[-1][:-4]+'.txt') ,'w') as save_f:
            tree = ET.parse(file)
            root = tree.getroot()
            objects = get_and_check(root,'objects',1)
            size = get_and_check(root,'size',1)
            height = float(get_and_check(size, 'height',1).text)
            width = float(get_and_check(size, 'width',1).text)
            for obj in get(objects,'object'):
                possibleresult = get_and_check(obj, 'possibleresult', 1)
                category = get_and_check(possibleresult, 'name', 1).text
                points = get_and_check(obj, 'points', 1)
                points = get(points, 'point')
                points = [pt.text.split(',') for pt in points][:4]
                pts = ''
                for pt in points:
                    pt[0] = clip(float(pt[0]),0.,width)
                    pt[1] = clip(float(pt[1]),0.,height)
                    pts += f"{pt[0]} {pt[1]} "
                save_f.write(pts + category.replace(' ','_') + ' 0\n')

def get_categories_count(srcpath):
    filelist = GetFileFromThisRootDir(srcpath)
    category_count = defaultdict(int)
    for file in filelist:
        print(file)
        tree = ET.parse(file)
        root = tree.getroot()
        objects = get_and_check(root,'objects',1)
        for obj in get(objects,'object'):
            possibleresult = get_and_check(obj, 'possibleresult', 1)
            category = get_and_check(possibleresult, 'name', 1).text
            category_count[category] += 1
    return category_count

def a(src,dstes):
    for dst in dstes:
        filelist = GetFileFromThisRootDir(dst,)
        for file in filelist:
            filename = os.path.split(file)[-1]
            shutil.copy(os.path.join(src,filename),file)

if __name__ == '__main__':
    xml2txt('/root/datasets/FAIR1M/train/labelXml/','/root/datasets/FAIR1M/train/labelTxt/')
    # category_count = get_categories_count(r'/private/datasets/FAIR1M2-raw/train/labelXml')
    # cc = list(category_count.items())
    # cc.sort(key=lambda x:x[0])
    # cc = list(zip(*cc))
    # with open("/private/datasets/FAIR1M2-raw/count.csv",'w') as f:
    #     f.write(",".join(map(str,cc[0])))
    #     f.write('\n')
    #     f.write(",".join(map(str,cc[1])))