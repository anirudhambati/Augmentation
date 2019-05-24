import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main_aug import *
from bbox_util import *


a = [[293,913, 2018, 1926, 4],[1147, 22, 1688, 893, 5],[42, 1963, 1538, 3120, 0],[3584, 2218, 4109, 2805, 9],[3759, 1730, 4160, 2309, 9],[2609, 534, 3159, 1238, 2],[2372, 1168, 2909, 1555, 8],[3159, 1030, 3634, 1380, 8],[2530, 1576, 3717, 2293, 6],[2072, 1272, 2422, 1709, 8],[1701, 655, 2155, 922, 3]]
img = cv2.imread("IMG_20190517_104211.jpg")[:,:,::-1]
bboxes = np.array(a)

plotted_img = draw_rect(img, bboxes)
plt.imshow(plotted_img)
plt.show()
'''
x = HorizontalFlip()
img_, bboxes_ = x(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

y = RandomRotate(5)
img3_, bboxes3_ = y(img.copy(), bboxes.copy())
plotted_img = draw_rect(img3_, bboxes3_)
plt.imshow(plotted_img)
plt.show()

z = RandomRotate(7.5)
img2_, bboxes2_ = z(img.copy(), bboxes.copy())
plotted_img = draw_rect(img2_, bboxes2_)
plt.imshow(plotted_img)
plt.show()
'''

x = RandomScale(0.5)
img_, bboxes_ = x(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()
