# TUWIEN - WS2020 CV: Task0 - Colorizing Images
# *********+++++++++*******++++INSERT GROUP NO. HERE
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import cv2
from typing import Tuple

def corr2d(img1: np.ndarray, img2: np.ndarray) -> float:
    # Returns the NCC of two given images.
    # Use the equation in the task description
    # img1 : first input image (n1 x m1 x 1)
    # img2 : second input image (n2 x m2 x 1)

    # student_code start
    mean = np.mean(img1)
    mean2 = np.mean(img2)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 != h2 or w1 != w2:
        #not necessarily a problem, but for this implementation might be
        raise ValueError("Arguments of different size") 

    nominator = 0
    tmp1 = img1 - mean
    tmp2 = img2 - mean2
    nominator = (tmp1 * tmp2).sum()

    tmp1 = (img1 - mean)**2
    tmp2 = (img2 - mean2)**2
    sum1 = tmp1.sum()
    sum2 = tmp2.sum()
    denominator = sqrt(sum1 * sum2)

    corr = nominator / denominator
    # student_code end
    # corr : correlation coefficient of given images (float)
    return corr

def _align_channel(channel: np.ndarray, alignTo: np.ndarray) -> Tuple[int, int]:
    # channel : image representing the channel to be rolled
    # alignTo : image representing the channel to which we want the best corr
    # Returns displaced channel

    best_x, best_y, best_corrG = 0, 0, 0
    org = np.copy(channel)
    for x_displacement in range(-15,15,1):
        tmp1 = np.roll(channel, x_displacement, axis=1)
        for y_displacement in range(-15,15,1):
            tmp2 = np.roll(tmp1, y_displacement, axis=0)
            tmp_corrG = corr2d(tmp2, alignTo)
            if tmp_corrG > best_corrG:
                best_x, best_y, best_corrG = x_displacement, y_displacement, tmp_corrG
    return np.roll(org, (best_x,best_y), axis=(1,0))

def align(imgR: np.ndarray, imgG: np.ndarray, imgB: np.ndarray) -> np.ndarray:
    # With the help of the NCC, align the color channels to colorize the image
    # HINT: np.roll(..)
    # imgR : image representing the red channel
    # imgG : image representing the green channel
    # imgB : image representing the blue channel

    # student_code start
    bestG = _align_channel(imgG, imgR)
    bestB = _align_channel(imgB, imgR)

    result = cv2.merge((imgR,bestG,bestB))
    # student_code end

    # result : colorized image (n x m x 3) - float
    return result
