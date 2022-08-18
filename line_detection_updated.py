'''
This module contains the functions and class that are used to detect
straight lines in astronomical images.

Process
---------
1. Ingest the raw image
2. Apply brightness cuts and standardize the image
3. Perform binary thresholding on the image
4. Blur the image
5. Find edges
6. Apply Hough line on the edge image
7. Cluster the estimated lines
8. Return all the estimated lines along with their cluster label

'''

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import logging

import cv2

from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import MeanShift

import yaml


def remove_background(img, sigma=3, maxiters=10, kernel_size=(70, 70), filter_size=(3, 3)):
    # Get background map and subtract.
    sigma_clip = SigmaClip(sigma, maxiters)
    median_bkg_estimator = MedianBackground()
    bkg = Background2D(img,
                       kernel_size,
                       filter_size=filter_size,
                       sigma_clip=sigma_clip,
                       bkg_estimator=median_bkg_estimator)
    background_map = bkg.background
    corrected = img - background_map

    return corrected


def image_mask(img, percent):
    """
    Provides mask coordinates for an image

    Parameters
    ----------
    img : `numpy.ndarray`
        The image to be masked
    percent : `float`
        percentage of the image borders to be masked out.
        If x percent is provided, x/2 percent of each edge is masked out

    Returns
    --------
    x_dim_left : `int`
        Left x coordinate of masking rectangle
    x_dim_right : `int`
        Right x coordinate of masking rectangle
    y_dim_top : `int`
        Top y coordinate of masking rectangle
    y_dim_bottom : `int`
        Bottom y coordinate of masking rectangle
    """
    x_dim_left = int(img.shape[0]*(percent/2))
    x_dim_right = int(img.shape[0]*(1-percent/2))
    y_dim_top = int(img.shape[1]*(percent/2))
    y_dim_bottom = int(img.shape[1]*(1-percent/2))

    return x_dim_left, x_dim_right, y_dim_top, y_dim_bottom


def show(img, ax=None, show=True, title=None, **kwargs):
    """
    Show image using matplotlib.

    Parameters
    ----------
    img : `np.array`
        Image to display.
    ax : `matplotlib.pyplot.Axes` or `None`, optional
        Ax on which to plot the image. If no axis is given
        a new figure is created.
    kwargs : `dict`, optional
        Keyword arguments that are passed to `imshow`.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if img.dtype == "uint8":
        ax.imshow(img, vmin=0, vmax=255, **kwargs)
    ax.imshow(img, **kwargs)

    ax.set_title(title, fontsize=22)

    return ax


def cluster(cart_coords, lines, bandwidth=50, plot_image=False):
    """
    Uses sklean's MeanShift to cluster the endpoint coordinates of detected lines

    Parameters
    ----------
    cart_coords : `list`
        List of tuples, each tuple containing a coordinate corresponding to
        the end point of a single estimated line
    lines : `numpy.ndarray`
        Polar coordinates of the detected lines
    bandwidth : `int`, optional, default=50
        Provides the bandwidth parameter for MeanShift algorithm
    plot_image : `bool`, optional, default=False
        Plots the assigned clusters

    Returns
    --------
    clustered_lines : `numpy.ndarray`
        An array where each row has the polar coordinates of an estimated line
        and the cluster label for that line

    See Also
    --------
    sklearn.cluster.MeanShift : MeanShift clustering algorithm
    """
    # Convert to an array
    cart_coords_array = np.array(cart_coords)

    # Apply MeanShift clustering
    clustering = MeanShift(bandwidth=bandwidth).fit(cart_coords_array)
    labels = clustering.labels_.reshape((len(clustering.labels_), 1))
    clustered_lines = np.hstack((lines, labels))

    if plot_image:
        # Plot the clusters
        for lab in set(clustering.labels_):
            mask = clustering.labels_ == lab
            x = cart_coords_array[:, 0]
            y = cart_coords_array[:, 1]

            plt.scatter(x[mask], y[mask], label=lab)
            plt.xlabel("x coordinate")
            plt.ylabel("y coordinate")
            plt.legend()

    return clustered_lines


# Create class
class LineDetection:
    '''
    Applies noise reduction image processing to the raw image, finds edges using Canny and
    detects straight lines using Scikit Image hough_line and hough_line_peaks

    Parameters
    ----------
    image : `numpy.ndarray`
        The raw image
    mask : `bool`, default=False
        Indicates whether or not to mask the image
    mask_percent : `float`, default=0.2
        The proportion of the image's edge to be masked
    brightness_cuts : `tuple`, default=(2,2)
        Limits of pixel brightness values, `(lower, upper)`, that will be cut.
        Pixels with brightness value smaller than `mean - lower * std`
        are set to 0, and pixel values larger than
        `mean + upper * std` are set to `mean + upper * std`
    thresholding_cut : `float`, default=0.5
        Applied on the processed image. The pixel intensity at these many standard
        deviations above the mean becomes the threshold limit. A binary threshold of
        0/255 is applied at the limit value
    threshold : `float`, default=0.075
        The voting threshold for Hough Line Peaks as a proportion of the length of
        the diagonal
    flux_prop_thresholds : `list`, default=[0.1,0.2,0.3,1]
        Lists the possible thresholds for proportion of bright pixels in the image,
        where a pixel is considered bright if it's intensity is above the mean
    blur_kernel_sizes : `list`, default=[3,5,9,11]
        Decides the kernel size for blurring the thresholded image and corresponds
        element wise to the flux_prop_thresholds list. Must be of the same length
        as flux_prop_thresholds
    '''
    # Instatiating constructors
    def __init__(self, image, **kwargs):
        # Assign the raw image
        self.image = image.copy()
        self.configure_from_dict(kwargs)

    def configure_from_dict(self, config):
        # Image processing parameters
        self.mask = config.pop("mask", False)
        self.mask_percent = config.pop("mask_percent", 0.2)
        self.brightness_cuts = config.pop("brightness_cuts", (2, 2))
        self.thresholding_cut = config.pop("thresholding_cut", 0.5)

        # Line detection parameters
        self.threshold = config.pop("threshold", 0.075)

        # Blurring parameters
        self.flux_prop_thresholds = config.pop("flux_prop_thresholds", [0.1, 0.2, 0.3, 1])
        self.blur_kernel_sizes = config.pop("blur_kernel_sizes", [3, 5, 9, 11])

    def configure_from_file(self, filepath):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        self.configure_from_dict(config)

    def process_image(self):
        '''
        Processes the raw image for hough_lines, using the following steps:
            1. Makes brightness cuts in the raw image
            2. Reduces the outlier values based on those brightness cuts
            3. Standardizes the image and fits it to 0-255 range
            4. Thresholds the image to 0 or 255 pixel values
            5. Performs optional masking of the edges
            6. Blurs the image
            7. Finds edges

        Returns
        -------
        thresholded_image : `numpy.ndarray`
            The image after applying brightness cuts and thresholding on
            the raw image
        blurred_image : `numpy.ndarray`
            The image after applying blurring to the thresholded image
        edges : `numpy.ndarray`
            The Canny edge detected image on which hough_lines would
            be applied

        See Also
        --------
        line_detection_updated.LineDetection.hough_transformation : Hough transformation function
        '''
        self.image = remove_background(self.image)
        trimmed_image = self.image.copy()

        # Making first brightness cuts in the image
        # Outliers on the positive side take the value of the cut.
        # Outliers on the negative side take the value 0
        up_limit = trimmed_image.mean() + self.brightness_cuts[1]*trimmed_image.std()
        low_limit = trimmed_image.mean() - self.brightness_cuts[0]*trimmed_image.std()
        trimmed_image[trimmed_image > up_limit] = up_limit
        trimmed_image[trimmed_image <= low_limit] = 0

        # Standardizing the image and moving it away from 0
        processed_image = (trimmed_image-trimmed_image.mean())/trimmed_image.std()
        processed_image -= processed_image.min()

        # Fitting the image to a 0-255 range
        processed_image = (255*(processed_image - processed_image.min())
                           / (processed_image.max() - processed_image.min()))

        # Thresholding the processed image
        limit = processed_image.mean() + self.thresholding_cut*processed_image.std()
        threshold, thresholded_image = cv2.threshold(processed_image, limit, 255, cv2.THRESH_BINARY)
        thresholded_image = cv2.convertScaleAbs(thresholded_image)

        # Masking image to remove any borders
        if self.mask is True:
            x_left, x_right, y_top, y_bottom = image_mask(thresholded_image, self.mask_percent)
            thresholded_image = thresholded_image[y_top:y_bottom, x_left:x_right]

        # Deciding Kernel size of blur based on amount of noise
        num_bright_pixels = np.sum(thresholded_image > np.mean(thresholded_image))
        x = thresholded_image.shape[0]
        y = thresholded_image.shape[1]
        prop_bright_pixels = num_bright_pixels/(x*y)

        for i in range(len(self.flux_prop_thresholds)):
            if prop_bright_pixels < self.flux_prop_thresholds[i]:
                kernel_blur = self.blur_kernel_sizes[i]
                break

        # Blur the image
        blurred_image = cv2.medianBlur(thresholded_image, kernel_blur)

        # Performing Canny edge detection
        edges = cv2.Canny(blurred_image, 0, 200)

        return (thresholded_image, blurred_image, edges)

    def hough_transformation(self):
        '''
        Detects the straight lines in an image

        Returns
        -------
        lines : `numpy.ndarray`
            The polar coordinates of detected lines
        angles_list : `list`
            List of angles (in radians) for each detected line
        cart_coords_list : `list`
            List of endpoint cartesian coordinates for each detected line
        thresholded_image : `numpy.ndarray`
            The thresholded image from process_image returns
        blurred_image : `numpy.ndarray`
            The blurred image from process_image returns
        edges : `numpy.ndarray`
            The Canny edge detected image from process_image returns

        See Also
        --------
        skimage.transform.hough_line : Scikit Image Hough Line function
        cv2.Canny : OpenCV Canny edge detector
        '''

        # Processing the image
        thresholded_image, blurred_image, edges = self.process_image()

        # Performing Hough transformation
        dimx, dimy = self.image.shape
        diagonal = np.sqrt(dimx**2 + dimy**2)
        thresh = int(self.threshold * diagonal)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

        # Applying Hough lines to retrieve all possible lines
        h, theta, d = hough_line(edges, theta=tested_angles)

        # Retriving the peaks
        accum, angles, dists = hough_line_peaks(h, theta, d, threshold=thresh)

        # Finding the cartesian coordinates and storing the returns
        lines = np.vstack((dists, angles)).T
        cart_coords_list = []
        angles_list = []

        logging.info(f"Number of detected lines = {len(dists)}")

        for i in range(len(angles)):
            (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            cart_coords_list.append((x0, y0))
            angles_list.append(angles[i])

        detection_dict = {"Lines": lines,
                          "Angles": angles_list,
                          "Cartesian Coordinates": cart_coords_list,
                          "Thresholded Image": thresholded_image,
                          "Blurred Image": blurred_image,
                          "Edges": edges}

        return detection_dict
