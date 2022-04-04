# TUWIEN - WS2020 CV: Task2 - Image Stitching
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
from matplotlib.pyplot import angle_spectrum
import numpy as np
# from cyvlfeat.sift.sift import sift
# import cyvlfeat
import os
import cv2


def get_panorama_data(path: str) -> Tuple[List[np.ndarray], List[cv2.KeyPoint], List[np.ndarray]]:
    # Loop through images in given folder, extract SIFT points
    # and return images, keypoints and descriptors
    # This time we need to work with color images. Since OpenCV uses BGR you need to swap the channels to RGB
    # HINT: os.walk(..), cv2.imread(..), sift=cv2.SIFT_create(), sift.detectAndCompute(..)
    #  path : path to image folder

    # student_code start
    img_data = []
    all_keypoints = []
    all_descriptors = []
    for root, dirs, files in os.walk(path, topdown = True):
        for f in files:
            print(f)
            im = cv2.imread(os.path.join(root, f))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img_data.append(im)
            im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            sift = cv2.SIFT_create()
            kp, desc = sift.detectAndCompute(im_gray, None)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            # kp_img=cv2.drawKeypoints(im_gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imwrite(f"results/kp_{f}", kp_img)
            
    # student_code end

    # img_data : list of images
    # all_keypoints : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    # all_descriptors : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    return img_data, all_keypoints, all_descriptors
