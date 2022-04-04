# TUWIEN - WS2020 CV: Task1 - Scale-Invariant Blob Detection
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import Tuple
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import utils


def create_log_kernel(size: int, sig: float) -> float:
    # Returns a rotationally symmetric Laplacian of Gaussian kernel
    # with given 'size' and standard deviation 'sig'
    # size : size of kernel (must be odd) (int)
    # sig : standard deviation (float)

    kernel = np.zeros((size, size), np.float)
    halfsize = int(np.floor(size / 2))
    r = range(-halfsize, halfsize+1, 1)
    for x in r:
        for y in r:
            hg = (np.power(np.float(x), 2) + np.power(np.float(y), 2)) / \
                (2 * np.power(np.float(sig), 2))
            kernel[x+halfsize, y+halfsize] = - \
                ((1.0 - hg) * np.exp(-hg)) / (np.pi * np.power(sig, 4))

    # kernel: filter kernel (size x size) (float)
    return kernel-np.mean(kernel)


def get_log_pyramid(org_img: np.ndarray, sigma: float, k: float, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    # Return a LoG scale space of given image 'img' with depth 'levels'
    # The filter parameter 'sigma' increases by factor 'k' per level
    # HINT: np.multiply(..), cv2.filter2D(..)
    # img : input image (n x m x 1) (float)
    # sigma : initial standard deviation for filter kernel (float)
    # levels : number of layers of pyramid (int)

    # student_code start
    images = []
    all_sigmas = []
    for level in range(1,levels):
        kernel_size = 2 * math.floor(sigma) + 1
        log_filter =  create_log_kernel(kernel_size, sigma)
        log_filter = log_filter * sigma**2
        img = cv2.filter2D(org_img, cv2.CV_32F, log_filter)
        img = abs(img)
        img /= np.max(img)
        all_sigmas.append(sigma)
        images.append(img)
        sigma = k * sigma
        # if level == 3: 
        utils.show_plot(img,"Group 11-13 12", f"task1_filtered_fromfilters_{level}.png")
    scale_space = np.stack(images,axis=2)

    # student_code end

    # scale_space : image pyramid (n x m x levels - float)
    # all_sigmas : standard deviation used for every level (levels x 1 - float)
    return scale_space, all_sigmas

