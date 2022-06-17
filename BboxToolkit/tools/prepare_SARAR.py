from collections import defaultdict
import cv2
import os
import shutil
from multiprocessing import Pool
import xml.etree.ElementTree as ET
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

def xml2txt(srcpath,dstpath):
    filelist = GetFileFromThisRootDir(srcpath)
    if not os.path.isdir(dstpath):
        os.mkdir(dstpath)
    for file in filelist:
        with open(os.path.join(dstpath,os.path.split(file)[-1][:-4]+'.txt') ,'w') as save_f:
            tree = ET.parse(file)
            root = tree.getroot()
            objects = get_and_check(root,'objects',1)
            for obj in get(objects,'object'):
                possibleresult = get_and_check(obj, 'possibleresult', 1)
                category = get_and_check(possibleresult, 'name', 1).text.replace(' ','_')
                points = get_and_check(obj, 'points', 1)
                points = get(points, 'point')[:4]
                pts = []
                for pt in points:
                    pts.extend([str(int(p)) for p in pt.text.split(',')])
                points = ' '.join(pts)
                save_f.write(points+' ' + category + ' 0\n')

def get_categories(srcpath,dstpath):
    filelist = GetFileFromThisRootDir(srcpath)
    if not os.path.isdir(dstpath):
        os.mkdir(dstpath)
    category_set = set()
    for file in filelist:
        tree = ET.parse(file)
        root = tree.getroot()
        objects = get_and_check(root,'objects',1)
        for obj in get(objects,'object'):
            possibleresult = get_and_check(obj, 'possibleresult', 1)
            category = get_and_check(possibleresult, 'name', 1).text
            category_set.add(category)
    return category_set
if __name__ == '__main__':
    srcpath = r'/private/datasets/sarar_bal_aug'
    xml2txt(os.path.join(srcpath,"gt"),os.path.join(srcpath,"labelTxt"))

def prepare_test(input_path):
    xml2txt(os.path.join(input_path,"gt"),os.path.join(input_path,"labelTxt"))
    