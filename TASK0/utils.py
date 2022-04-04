# TUWIEN - WS2020 CV: Task0 - Colorizing Images
import matplotlib.pyplot as plt
import numpy as np


def show_plot(img: np.ndarray, group_no: str=None, name: str=None):
    # Plots the given image'img', use additional arguments to
    # save the image to folder "results" in your local directory with given name
    # img : images to save
    # group_no : your group number (optional)
    # name : name of saved file (optional)

    fig = plt.figure()
    plt.imshow(img)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/'+name)

    plt.plot()
