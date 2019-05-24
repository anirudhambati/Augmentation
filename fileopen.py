import os
import numpy as np
from main_aug import *
from bbox_util import *
import cv2
import matplotlib.pyplot as plt

path = '/Users/ambatianirudh/Downloads/testee/'

def Convert(string):
    numbers = list(string.split(" "))
    numbers = [ int(float(x)) for x in numbers ]
    return numbers

# Read every file in directory
for filename in os.listdir(path):

    if filename.endswith(".txt"):

        try:
            fpath = path + filename
            file = open(fpath, 'r')
            a = []
                # Read each line of the file
            for line in file:
                a.append(Convert(line))

            bboxes = np.array(a)
            i = filename[:-4] + '.jpg'
            img = cv2.imread(path + i)#[:,:,::-1]

            #### AUGMENTATION HERE ####

            x = RandomScale(0.5)
            img_, bboxes_ = x(img.copy(), bboxes.copy())
            f = open('/Users/ambatianirudh/Downloads/res_testee/scale/' + 'rscale' + filename, 'w+')
            for j in bboxes_.tolist():
                temp = " ".join(str(x) for x in j)
                f.write( temp + '\n')

            cv2.imwrite('/Users/ambatianirudh/Downloads/res_testee/scale/' + 'rscale' + i, img_)

            ###########################

        except:
            print('File not found -- skipped \n')
            continue

    else:
        continue
