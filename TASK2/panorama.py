# TUWIEN - WS2020 CV: Task2 - Image Stitching
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    # Stitch the final panorama with the calculated panorama extents
    # by transforming every image to the same coordinate system as the center image. Use the dot product
    # of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    # HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    # images : list of images
    # width : width of panorama (in pixel)
    # height : height of panorama (in pixel)
    # H : list of homographies to center image ([number_of_images x 3 x 3])
    # T : translation matrix for panorama ([3 x 3])

    # student_code start
    result = np.zeros((height, width,3), dtype=np.uint8)
    T = np.float32(T)
    transforms = [T @ hm for hm in H]
    for im_id in range(len(images)):
        out = cv2.warpPerspective(images[im_id], transforms[im_id], (width, height))
        result = cv2.addWeighted(result, 1, out, 0.4, 0)

    # student_code end
    # result : panorama image ([height x width x 3])
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    # Use the equation from the assignment description to overlay transformed
    # images by blending the overlapping colors with the respective alpha values
    # HINT: ndimage.distance_transform_edt(..)
    # images : list of images
    # width : width of panorama (in pixel)
    # height : height of panorama (in pixel)
    # H : list of homographies to center image ([number_of_images x 3 x 3])
    # T : translation matrix for panorama ([3 x 3])

    # student_code start

    # student_code end
    # result : blended panorama image ([height x width x 3])

    return result
