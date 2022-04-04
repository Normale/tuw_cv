# TUWIEN - WS2020 CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++Group_9-11_11

from typing import List, Tuple
import numpy as np
import cv2
import os


class SceneDataset():
    images = []         # list of images
    labels = []         # list of labels of images
    class_names = []    # list with of class names (folder names)

    def __init__(self, path: str, maxDataPerLabel:int = None, resize:bool = False):
        # Loop through all subfolders within the given 'path', get all images per folder,
        # save the images in gray scale and normalize the image values between 0 and 1.
        # The label of an image is the current subfolder (eg. value between 0-9 when using 10 classes)
        # HINT: os.walk(..), cv2.imread(..)
        # path : path to dataset - string

        # student_code start
        img_data = []
        labels = []
        dirs = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            for label, classname in enumerate(dirnames):
                dirs.append(classname)
                for (dirpath, dirnames, filenames) in os.walk(path+os.sep+classname):
                    for i, imgName in enumerate(filenames):
                        if(maxDataPerLabel == None or i < maxDataPerLabel):
                            img = cv2.imread(dirpath+os.sep+imgName)
                            if resize:
                                img = cv2.resize(img,(100,100))
                            img_data.append(img)
                            labels.append(label)

        # student_code end

        # save as local parameters:
        # images : list of images [num_of_images x n x m] - float
        # labels : list of belonging label per image [num_of_images x 1] - int
        # class_names : list of names of the subfolders (classes) [num_of_classes x 1] - string
        self.images = img_data
        self.labels = labels
        self.class_names = dirs

    def get_data(self) -> Tuple[List[np.ndarray], List[int]]:
        return self.images, self.labels

    def get_class_names(self) -> List[str]:
        return self.class_names
