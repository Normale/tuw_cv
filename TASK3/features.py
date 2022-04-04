from typing import List
from sklearn.metrics import pairwise_distances
import cv2
import numpy as np
import random
import time
from tqdm.notebook import tqdm

def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    # Extract dense feature points on a regular grid with 'stepsize' and if given, return
    # 'num_samples' random samples per image. if 'num_samples' is not given, take all features
    # extracted with the given 'stepsize'. sift.compute has the argument "keypoints", set it to
    # a list of keypoints for each square.
    # HINT: cv2.KeyPoint(...), cv2.SIFT_create(), sift.compute(img, keypoints), random.sample(..)
    # images : list of images to extract dsift [num_of_images x n x m] - float
    # stepsize : grid spacing, step size in x and y direction - int
    # num_samples : random number of samples per image (optional) - int

    tic = time.perf_counter()

    # student_code start
    if num_samples == None:
        num_samples = 100

    # TODO do we need to init sift for each image new?
    sift = cv2.SIFT_create()


    all_descriptors = []
    for image in tqdm(images):
        height,width,channels = np.shape(image)
        keypoints = []
        for x in range(0,width,stepsize):
            for y in range(0,height,stepsize):
                keypoints.append(cv2.KeyPoint(x,y,1))
        selected_keypoints = random.sample(keypoints,num_samples)

        keypoints, descriptors = sift.compute(image,selected_keypoints)
        all_descriptors.append(descriptors)

    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:",  {toc - tic}, " seconds")

    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    # For classification, generate a histogram of word occurence per image
    # Use sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    # to the nearest centroids and count the occurences of each centroids. The histogram
    # should be as long as the vocabulary size (number of centroids)
    # HINT: sklearn_pairwise.pairwise_distances(..), np.histogram(..)
    # dense_feat : list sift descriptors per image [number_of_images x num_samples x 128] - float
    # centroids : centroids of clusters [vocabulary_size x 128]

    tic = time.perf_counter()

    # student_code start
    histograms = []
    bin_edges = []

    for image in tqdm(dense_feat):
        clusters = []
        for sample in image:
            distances = []
            for centroid in centroids:
                # I had to do this stupid loop, because centroids can not be turned into
                # nd-array, since max ndarray dimensions is 32.
                # There probably is better solution
                # so instead of checking pairwise distance of matrices, I check the pairwise
                # distance of each vector 
                # ("vector" means either 128-dimensions 128-dimensional feature) 
                # with each centroid (128-dim as well) and make list of distances 
                s = sample.reshape(1, -1)
                c = centroid.reshape(1, -1)
                distance = pairwise_distances(
                    s, c, metric="euclidean"
                )
                distances.append(distance)
            closest_centroid_id = np.argmin(distances)
            # print("Smallest distance to cluster nr.", closest_centroid_id)
            clusters.append(closest_centroid_id)
        histogram, bin_edge = np.histogram(clusters, bins=len(centroids)+1, range=(0,len(centroids)+1))
        histograms.append(histogram)
        bin_edges.append(bin_edge)
        #break here if you dont want it to run too long
        #break
    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:",  {toc - tic}, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms, bin_edges
