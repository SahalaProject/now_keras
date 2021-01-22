# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import sys
import shutil

"""
转换图片为TF使用格式
"""

# validation  training


def turnto24(input_path, out_path):
   files = os.listdir(input_path)
   files = np.sort(files)
   i=0
   for f in files:
       if f != '.DS_Store':
           imgpath = input_path + f
           print(imgpath)
           try:
               img = Image.open(imgpath).convert('RGB')
               dirpath = out_path
               file_name, file_extend = os.path.splitext(f)
               dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
               img.save(dst)
           except:
                os.remove(imgpath)
                print('remove...', imgpath)
       else:
           os.remove(input_path + f)

if __name__ == '__main__':
    name_list = [ '美国卷毛猫', '欧洲缅甸猫']

    for name in name_list:
        path = r'D:\11111111111111111\官方-demo\download/{}/'.format(name)
        input_path = path
        out_path = path

        turnto24(input_path, out_path)
