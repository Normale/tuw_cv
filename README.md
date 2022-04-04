# tuw_cv


# Parts (click on Task to see .ipynb results):


## [Task 0: Aligning 3 images based on normalized cross-correlation (NCC).](https://github.com/Normale/tuw_cv/blob/main/TASK0/Task0.ipynb):
- Implementation of NCC in numpy
- aligning images based on correlation

![]()
![]()
  <p align="center">
  <img width="30%" height="30%" src="https://i.imgur.com/OApfRGJ.png">
</p>
  <p align="center">
  <img width="70%" height="70%" src="https://i.imgur.com/srRX7BG.png">
</p>



## [Task 1: Scale-Invariant Blob Detection](https://github.com/Normale/tuw_cv/blob/main/TASK1/Task1.ipynb):
 - implementation of scale-normalized Laplacian of Gaussian (LoG) operator
 - building LoG pyramid
 - finding blobs from local maxes
![]()
  <p align="center">
  <img width="50%" height="50%" src="https://i.imgur.com/pFSSz6v.png">
</p>




## [Task 2: Image Stitching](https://github.com/Normale/tuw_cv/blob/main/TASK2/Task2.ipynb):
 - detecting keypoints using SIFT
 - estimation of putative matches between local descriptors of 2 images
 - usage of RANSAC (because of false-matches) to estimate transformation between images
 - join image based on transformation matrices
  <p align="center">
  <img width="50%" height="50%" src="https://i.imgur.com/KUSKt6H.png">
</p>
 
 ## [Task 3: Classical ML approach to image classification](https://github.com/Normale/tuw_cv/blob/main/TASK2/Task2.ipynb):
 - extract features from images 
 - create clusters of image features (use KMeans to create bag-of-words model)
 - use k-NN to predict class based on manually extracted features
  <p align="center">
  <img  src="https://i.imgur.com/Ue21ZHr.png">
</p>

 
  
 ## [Task 4: Deep Learning approach to image classification](https://github.com/Normale/tuw_cv/blob/main/TASK2/Task2.ipynb):
 - Implementation of PyTorch model to classify images
     - Dataset (augmentation, loading)
     - Trainer (losses, optimizers, training in general)
     - Classifier (NN architecture)

 
 
 
 
