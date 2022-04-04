# TUWIEN - WS2020 CV: Task2 - Image Stitching
from typing import List
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint], group_no: str = None, name: str = None) -> None:
    # Plots the given 'image' and 'keypoints' on top
    # Use additional arguments to save the image to folder "results" in your local directory with given name
    # image : image to plot
    # keypoints : keypoints of given image ([number_of_keypoints] - KeyPoint)
    # group_no : your group number (optional)
    # name : filename (without extension) (optional)

    fig, ax = plt.subplots()

    image = cv2.drawKeypoints(image, keypoints, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(image)

    # for kp in keypoints:
    #     circ = Wedge((kp[1],kp[0]),kp[2], kp[3]*180/np.pi, (kp[3]*180/np.pi)-10,color='y',fill=False)
    #     ax.add_patch(circ)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)

    # Show the image
    plt.show()


def plot_matches(img1: np.ndarray, img2: np.ndarray, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], group_no: str = None, name: str = None) -> None:
    # Plots both images next to each other and a connection line between image keypoints (matches)
    # Use additional arguments to save the image to folder "results" in your local directory with given 'name'
    # img1 : left image
    # img2 : right image
    # kp1 : keypoints left image ([number_of_keypoints] - KeyPoint)
    # kp2 : keypoints right image ([number_of_keypoints] - KeyPoint)
    # matches : indices matching keypoints([number_of_matches x 2] - int)
    # group_no : your group number (optional)
    # name : filename (without extension) (optional)

    fig, (ax1) = plt.subplots(1, 1)
    result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [
                                [m] for m in matches], None)
    ax1.imshow(result)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)

    # Show the image
    plt.show()


def plot_transformed_image(img1: np.ndarray, img2: np.ndarray, group_no: str = None, name: str = None) -> None:
    # Plots an alpha composition of two image
    # Use additional arguments to save the image to folder "results" in your local directory with given name
    # img1 : left image
    # img2 : right image
    # group_no : your group number (optional)
    # name : filename(without extension)  (optional)

    added_image = cv2.addWeighted(img1, 1, img2, 1, 0)

    fig, ax = plt.subplots()
    ax.imshow(added_image)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)

    plt.show()


def plot_panorama(img: np.ndarray, group_no: str = None, name: str = None) -> None:
    # Plots an image
    # Use additional arguments to save the image to folder "results" in your local directory with given name
    # img : panorama image
    # group_no : your group number (optional)
    # name : name of saved file (without extension) (optional)

    fig, ax = plt.subplots()
    ax.imshow(img)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)

    plt.show()
