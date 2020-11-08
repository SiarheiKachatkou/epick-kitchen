import cv2
import numpy as np


def flip(image):
    return cv2.flip(image,1)


def shift_left(image, shift=0.05):
    height,width=image.shape[:2]
    shiftMat=np.float32([[1,0,int(shift*width)],[0,1,0]])
    image = cv2.warpAffine(image, shiftMat, (width, height), cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    return image


def shift_right(image, shift=0.05):
    height,width=image.shape[:2]
    shiftMat = np.float32([[1, 0, -int(shift * width)], [0, 1, 0]])
    image = cv2.warpAffine(image, shiftMat, (width, height), cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    return image


def ident(image):
    return image

def get_4_augms_list():
    return [flip, shift_left, shift_right, ident]


def get_1_augms_list():
    return [ident]
