# TUWIEN - WS2020 CV: Task2 - Image Stitching
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2
from math import sqrt

def get_geometric_transform(p1, p2):
    # Calculate a homography from the first set of points (p1) to the second (p2)
    # p1 : first set of points
    # p2 : second set of points
    num_points = len(p1)
    A = np.zeros((2*num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2*p] = np.concatenate(([0, 0, 0], -first, p2[p, 1]*first))
        A[2*p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0]*first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    # Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    # HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)
    # kp1 : keypoints left image ([number_of_keypoints] - KeyPoint)
    # kp2 : keypoints right image ([number_of_keypoints] - KeyPoint)
    # matches : indices of matching keypoints ([number_of_matches] - DMatch)

    # student_code start
    points_1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points_1 = points_1.reshape((1,) + points_1.shape)
    points_2 = np.array([kp2[m.trainIdx].pt for m in matches])
    points_2 = points_2.reshape((1,) + points_2.shape)
    best_in_count = 0
    # best_transform = None
    best_inliers = []
    for i in range(100):
        inliers = []
        inlier_count = 0
        r_matches = random.sample(matches, 4)
        s1 = np.array([kp1[m.queryIdx].pt for m in r_matches])
        s2 = np.array([kp2[m.trainIdx].pt for m in r_matches])
        g_trans = get_geometric_transform(s1, s2)
        # s1 = s1.reshape(1,4,2)
        # s2 = s2.reshape(1,4,2)
        p_transform = cv2.perspectiveTransform(points_1, g_trans)
        # print(points_2)
        # print(f"{p_transform}")
        for i in range(len(matches) - 1):
            euc = np.linalg.norm(p_transform[0][i] -  points_2[0][i])
            # print(f"{euc=}")
            if euc < 5: 
                inlier_count += 1
                inliers.append(i)
        if inlier_count > best_in_count:
            best_in_count = inlier_count
            # best_transform = g_trans
            best_inliers = inliers
    
    inliers_points = np.array([points_1[0][i] for i in best_inliers])
    inliers_points_2 = np.array([points_2[0][i] for i in best_inliers])
    trans = get_geometric_transform(inliers_points, inliers_points_2)
    inliers = best_inliers #just name change

    # # test
    # inlier_count = 0
    # p_transform = cv2.perspectiveTransform(points_1, homography)
    # for i in range(len(matches) - 1):
    #     euc = np.linalg.norm(p_transform[0][i] - points_2[0][i])
    #     # print(f"{euc=}")
    #     if euc < 5: 
    #         inlier_count += 1
    # print(f"new inlier count = {inlier_count}")
    # student_code end

    # trans : homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    # inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    return trans, inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    # Prepare all homographies by calculating the transforms from all other images
    # to the reference image of the panorama (center image)
    # First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    # two consecutive images from left to right, then calculate and return the homographies to the center image
    # HINT: inv(..)
    # desc : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    # kp : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    # center_id : id of center image of panorama

    # student_code start
    homographies = []
    for i in range(len(desc) - 1):
        matches = mapping.calculate_matches(desc[i], desc[i+1])
        transform_M, inliers_indexes = get_transform(kp[i], kp[i+1], matches)
        homographies.append(transform_M)
    h12, h23, h34, h45 = homographies
    h43 = np.linalg.inv(h34)
    h54 = np.linalg.inv(h45)
    h13 = h12 @ h23
    h53 = h54 @ h43
    h33 = np.identity(3, dtype=np.float32)
    H_center = [h13, h23, h33, h43, h53]

    # for i in range(len(homographies)):
    #     center_id = 1 + np.floor(len(homographies) // 2)
    #     start = i if i < center_id else center_id
    #     step = 1 if i < center_id else -1
    #     stop = (center_id + 1) if step > 0 else (center_id - 1)
    #     h_temp = homographies[start]
    #     for j in range(start, stop, step):
    #         print(f"homography {i}{j}, {step=} {start=} {stop=}")
    #         homography
    #         h_temp = np.matmul(h_temp, homographies[j])
    #     H_center.append(h_temp)
    # student_code end

    # H_center : list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    # Calculate the extent of the panorama by transforming the corners of every image
    # and get the minimum and maxima in x and y direction, as you read in the assignment description.
    # Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    # panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    # HINT: cv2.perspectiveTransform(..)
    # images : list of images
    # H : list of homographies to center image ([number_of_images x 3 x 3])

    # student_code start
    y,x,_ = images[0].shape #assumption images are the same size
    corners = [(0,0), (0,x-1), (0,y-1), (x-1,y-1)]
    corners_all = []
    for homography in H:   
        transformed = []
        for x,y in corners:
            pt = np.array([[[x,y]]], dtype = "float32")
            res = cv2.perspectiveTransform(pt, homography)
            transformed.append(res.reshape(2))
        corners_all.append(transformed)
    ar = np.array(corners_all).reshape(20,2)
    xs = ar[:,0]
    ys = ar[:,1]
    xmax = np.ceil(xs.max())
    ymax = np.ceil(ys.max())
    xmin = np.floor(xs.min())
    ymin = np.floor(ys.min())
    width = xmax - xmin
    height = ymax - ymin
    # print(width, height)
    T = np.array([[1,0,np.abs(xmin)],[0,1,np.abs(ymin)],[0, 0, 1]])
    # student_code end

    # T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    # width : width of panorama (in pixel)
    # height : height of panorama (in pixel)
    return T, int(width), int(height)
