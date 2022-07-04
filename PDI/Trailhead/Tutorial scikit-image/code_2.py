import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.segmentation import chan_vese
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb

astronaut = data.astronaut()
gray_astronaut = rgb2gray(astronaut)

###Segmentação de Chan-Vese###
fig, axes = plt.subplots(1, 3, figsize=(10, 10))
# Sample Image of scikit-image package
# Computing the Chan VESE segmentation technique
chanvese_gray_astronaut = chan_vese(gray_astronaut,
max_iter=100,
extended_output=True)
ax = axes.flatten()
# Plotting the original image
ax[0].imshow(gray_astronaut, cmap="gray")
ax[0].set_title("Original Image")
# Plotting the segmented - 100 iterations image
ax[1].imshow(chanvese_gray_astronaut[0], cmap="gray")
title = "Chan-Vese segmentation - {} iterations".format(len(chanvese_gray_astronaut[2]))
ax[1].set_title(title)
# Plotting the final level set
ax[2].imshow(chanvese_gray_astronaut[1], cmap="gray")
ax[2].set_title("Final Level Set")
plt.show()

###Segmentação Não Supervisionada###

# Setting the plot figure as 15, 15
plt.figure(figsize=(15, 15))
# Applying SLIC segmentation
# for the edges to be drawn over
astronaut_segments = slic(astronaut,n_segments=100,compactness=1)
plt.subplot(1, 2, 1)
# Plotting the original image
plt.imshow(astronaut)
# Detecting boundaries for labels
plt.subplot(1, 2, 2)
# Plotting the ouput of marked_boundaries
# function i.e. the image with segmented boundaries
plt.imshow(mark_boundaries(astronaut, astronaut_segments))
plt.show()

###Limirização com Clusterização Iterativa###

# Setting the plot size as 15, 15
plt.figure(figsize=(15,15))
# Applying Simple Linear Iterative
# Clustering on the image
# - 50 segments & compactness = 10
astronaut_segments = slic(astronaut,n_segments=50,compactness=10)
plt.subplot(1,2,1)
# Plotting the original image
plt.imshow(astronaut)
plt.subplot(1,2,2)
# Converts a label image into
# an RGB color image for visualizing
# the labeled regions.
plt.imshow(label2rgb(astronaut_segments,astronaut,kind = 'avg'))
plt.show()


###Segmentação de Felzenszwalb###
# Importing the required libraries

# Setting the figure size as 15, 15
plt.figure(figsize=(15,15))
# computing the Felzenszwalb's
# Segmentation with sigma = 5 and minimum
# size = 100
astronaut_segments = felzenszwalb(astronaut,scale = 2,sigma=5,min_size=100)
# Plotting the original image
plt.subplot(1,2,1)
plt.imshow(astronaut)
# Marking the boundaries of
# Felzenszwalb's segmentations
plt.subplot(1,2,2)
plt.imshow(mark_boundaries(astronaut,astronaut_segments))
plt.show()