# TUWIEN - WS2020 CV: Task1 - Scale-Invariant Blob Detection
# *********+++++++++*******++++INSERT GROUP NO. HERE
import matplotlib.pyplot as plt
import numpy as np
import math

def show_plot(img: np.ndarray, group_no: str = None, name: str = None) -> None:
    # Plots the given image 'img', use additional arguments to
    # save the image to folder "results" in your local directory with given name
    # img: images to save
    # group_no: your group number - string (optional)
    # name: filename (without extension) - string (optional)

    fig = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.plot()

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)


def show_blobs(img: np.ndarray, peaks: np.ndarray, all_sigmas: np.ndarray, group_no: str=None, name: str=None):
    # Plots 'img' with circles at locations given with 'peaks' and radius based on level where maximum of the peak, can be
    # obtained using 'all_sigmas' and the equation in the assignment description to calculate the radii per peak location.
    # Use additional arguments to save the image to folder "results" in your local directory with given 'name'
    # HINT: np.take(..)
    # img : input image (n x m x 1 - float)
    # peaks : matrix with peak coordinates (row, column, level) [number_of_maxima x 3]
    # all_sigmas : list of sigma values per level (num_levels x 1 - float)
    # group_no : your group number - string (optional)
    # name : filename (without extension) - string (optional)

    # student_code start
    radi = [(peak[2]+1) * math.sqrt(2) for peak in peaks]
    # radi = [r if r > 8 else 0 for r in radi ]
    # student_code end

    # do not change from here
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img, cmap="gray")

    for i in range(len(peaks)):

        circ = plt.Circle((peaks[i, 1], peaks[i, 0]),
                          radi[i], color='r', fill=False)
        ax.add_patch(circ)

    fig.suptitle("blobs: " + str(len(peaks)))

    if name is not None and group_no is not None:
        fig.suptitle(group_no + ", blobs: " + str(len(peaks)))
        plt.savefig('results/'+name)

    # Show the image
    plt.show()
