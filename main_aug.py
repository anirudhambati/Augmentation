import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from bbox_util import *

class HorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] + 2*(img_center[[0, 2]] - bboxes[:, [0, 2]]))

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes

'''
class Rotate(object):
    """Rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated


    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle):
        self.angle = angle


    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """

        angle = self.angle
        print(self.angle)

        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        img = rotate_im(img, angle)
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w,h))
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes
'''


class RandomRotate(object):

    def __init__(self, angle):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):

        angle = random.uniform(*self.angle)
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        img = rotate_im(img, angle)
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w,h))
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes


class RandomScale(object):
    """Randomly scales an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale = 0.2, diff = False):
        self.scale = scale


        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff



    def __call__(self, img, bboxes):


        #Chose a random digit to scale by

        img_shape = img.shape

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x



        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

        bboxes[:,:4] = (bboxes[:,:4] * [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y])



        canvas = np.zeros(img_shape, dtype = np.uint8)

        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])


        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]

        img = canvas
        bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)


        return img, bboxes


class Scale(object):
    """Scales the image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.


    Parameters
    ----------
    scale_x: float
        The factor by which the image is scaled horizontally

    scale_y: float
        The factor by which the image is scaled vertically

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale_x = 0.2, scale_y = 0.2):
        self.scale_x = scale_x
        self.scale_y = scale_y


    def __call__(self, img, bboxes):


        #Chose a random digit to scale by

        img_shape = img.shape


        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y

        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

        bboxes[:,:4] = (bboxes[:,:4] * [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y])



        canvas = np.zeros(img_shape, dtype = np.uint8)

        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])


        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]

        img = canvas
        bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)


        return img, bboxes
