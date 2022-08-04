'''
This module creates the functions and class that are used to detect
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

import cv2

from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import MeanShift

# Defining functions


def image_mask(img, percent):
    """
    Provides mask coordinates for an image

    Parameters
    ----------
    img : 'numpy.ndarray'
        The image to be masked
    percent : `float`
        percentage of the image borders to be masked out.
        If x percent is provided, x/2 percent of each edge is masked out

    Returns
    --------
    mask coordinates : `int`
        coordinates of the numpy array at which masking should be applied
    """
    x_dim_top = int(img.shape[0]*(percent/2))
    x_dim_bottom = int(img.shape[0]*(1-percent/2))
    y_dim_left = int(img.shape[1]*(percent/2))
    y_dim_right = int(img.shape[1]*(1-percent/2))

    return x_dim_top, x_dim_bottom, y_dim_left, y_dim_right


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

    # make the plots bigger
    plt.rcParams["figure.figsize"] = (10, 10)

    if ax is None:
        fig, ax = plt.subplots()

    if "uint8" == img.dtype:
        ax.imshow(img, vmin=0, vmax=255, **kwargs)
    ax.imshow(img, **kwargs)

    ax.set_title(title, fontsize=22)

    return ax


def cluster(cart_coords, lines, bandwidth=50):
    """
    Uses sklean's MeanShift to cluster the endpoint coordinates of detected lines

    Parameters
    ----------
    cart_coords : 'list'
        List of tuples, each tuple containing a coordinate corresponding to
        the end point of a single estimated line
    lines : `numpy.ndarray`
        Polar coordinates of the detected lines
    bandwidth: 'int', optional, default=50
        Provides the bandwidth parameter for MeanShift algorithm

    Returns
    --------
    clusteredLines : `numpy.ndarray`
        An array where each row has the polar coordinates of an estimated line
        and the cluster label for that line
    """
    # Convert to an array
    cart_coords_array = np.array(cart_coords)

    # Apply MeanShift clustering
    clustering = MeanShift(bandwidth=bandwidth).fit(cart_coords_array)
    labels = (clustering.labels_).reshape((len(clustering.labels_), 1))
    clustered_lines = np.hstack((lines, labels))

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
    '''

    # Instatiating constructors
    def __init__(self):
        '''
        Initializes the class' parameters

        Parameters
        ----------
        image : 'numpy.ndarray'
            The raw image
        mask : `bool`, default=False
            Indicates whether or not to mask the image
        mask_percent: 'float', default=0.2
            The proportion of the image's edge to be masked
        nstd1_cut: 'float', default=2
            Applied on the raw image. Pixel intensity above these many standard deviations are
            reduced to the intensity value at that standard deviation. Pixels below these many
            standard deviations are converted to 0
        nstd2_binary_cut: 'float', default=0.5
            Applied on the processed image. The pixel intensity at these many standard
            deviations above the mean becomes the threshold limit. A binary threshold of
            0/255 is applied at the limit value
        threshold: 'float', default=0.075
            The voting threshold for Hough Line Peaks as a proportion of the length of
            the diagonal
        flux_prop_thresholds: 'list', default=[0.1,0.2,0.3,1]
            Lists the possible thresholds for proportion of bright pixels in the image,
            where a pixel is considered bright if it's intensity is above the mean
        blur_kernel_sizes: 'list', default=[3,5,9,11]
            Decides the kernel size for blurring the thresholded image and corresponds
            element wise to the flux_prop_thresholds list. Must be of the same length
            as flux_prop_thresholds
        '''
        # Assign the raw image
        self.image = None

        # Image processing parameters
        self.mask = False
        self.mask_percent = 0.2
        self.nstd1_cut = 2
        self.nstd2_binary_cut = 0.5

        # Line detection parameters
        self.threshold = 0.075

        # Blurring parameters
        self.flux_prop_thresholds = [0.1, 0.2, 0.3, 1]
        self.blur_kernel_sizes = [3, 5, 9, 11]

    def process_image(self):
        '''
        Processes the raw image for hough_lines

        Parameters
        ----------
        self : 'class'
            The LineDetection class to use its __init__ parameters

        Returns
        -------
        thresholded_image: 'numpy.ndarray'
            The image after applying brightness cuts and thresholding on
            the raw image
        blurred_image: 'numpy.ndarray'
            The image after applying blurring to the thresholded image
        edges: 'numpy.ndarray'
            The Canny edge detected image on which hough_lines would
            be applied
        '''
        trimmed_image = self.image

        # Making first cuts in the image
        # Outliers on the positive side take the value of the cut.
        # Outliers on the negative side take the value 0
        up_limit = trimmed_image.mean() + self.nstd1_cut*trimmed_image.std()
        low_limit = trimmed_image.mean() - self.nstd1_cut*trimmed_image.std()
        trimmed_image[trimmed_image > up_limit] = up_limit
        trimmed_image[trimmed_image <= low_limit] = 0

        # Standardizing the image and moving it away from 0
        processed_image = (trimmed_image-trimmed_image.mean())/trimmed_image.std()
        processed_image -= processed_image.min()

        # Fitting the image to a 0-255 range
        processed_image = (255*(processed_image - processed_image.min())
                           / (processed_image.max() - processed_image.min()))

        # Thresholding the processed image
        limit = processed_image.mean() + self.nstd2_binary_cut*processed_image.std()
        threshold, thresholded_image = cv2.threshold(processed_image, limit, 255, cv2.THRESH_BINARY)
        thresholded_image = cv2.convertScaleAbs(thresholded_image)

        # Masking image to remove any borders
        if self.mask is True:
            x_top, x_bottom, y_left, y_right = image_mask(thresholded_image, self.mask_percent)

            thresholded_image = thresholded_image[x_top:x_bottom, y_left:y_right]

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

        Parameters
        ----------
        self : 'class'
            The LineDetection class to use its __init__ parameters

        Returns
        -------
        lines: 'numpy.ndarray'
            The polar coordinates of detected lines
        angles_list: 'list'
            List of angles (in radians) for each detected line
        cart_coords_list: 'list'
            List of endpoint cartesian coordinates for each detected line
        thresholded_image: 'numpy.ndarray'
            The thresholded image from process_image returns
        blurred_image: 'numpy.ndarray'
            The blurred image from process_image returns
        edges: 'numpy.ndarray'
            The Canny edge detected image from process_image returns
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
        print(f"Found {len(dists)} lines.")

        # Finding the cartesian coordinates and storing the returns
        lines = np.vstack((dists, angles)).T
        cart_coords_list = []
        angles_list = []

        for i in range(len(angles)):
            (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            cart_coords_list.append((x0, y0))
            angles_list.append(angles[i])

        return lines, angles_list, cart_coords_list, thresholded_image, \
            blurred_image, edges
