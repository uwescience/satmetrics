'''
Author: Ashley Santos

This module contains pixel plotting module that can be used to equalize and plot the median intensity values of the rotated image.
Use the line_detection_testing.ipynb file to import this module and apply it on multiple images.

Process (in development currently):
    1. Declare init variables (see description of parameters within the class)
    2. Choose image
    3. Process the image:
        a. Apply z_score_trim once to reduce the outlier pixel values
        b. Apply z_score_trim on this processed image again
        c. Standardize and normalize the image using cv2
        d. Perform Canny edge detection
        e. Trim the edges of the image (if chosen)
    4. Perform Hough Transformation on the Canny edge image to get streak coordinates
    5. Convert streak polar coordinates into cartesian coordinates
    6. Find the mean coordinates at the edges of the image
    7. Determine the angle of rotation for the image based on the mean coordinates
    8. Rotate the image at the determined angle of rotation
    9. Create plot of median pixel intensity values utilizing the rotated image 
 

Next steps from here:
    Quantify brightness and plot a histogram of brightness of the image
'''

import line_detection
import image_rotation
from astropy.io import fits
import astropy.visualization as aviz
import numpy as np
import matplotlib.pyplot as plt




def pixelplot(rotated_image, title, detector):
    stretch = aviz.HistEqStretch(rotated_image)
    norm = aviz.ImageNormalize(rotated_image, stretch=stretch, clip=True)
    histeq = norm(rotated_image)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    detector.show(histeq, ax=ax1, title=title)
    ax1.set_ylabel("Pixel", fontsize=18)
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(212)
    plt.plot(np.median(rotated_image, axis= 1))
    ax2.set_xlabel("Pixel", fontsize=16)
    ax2.set_ylabel("Counts/pixel", fontsize=18)





