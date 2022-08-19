"""
Author: Kilando Chambers

This module contains multiple functions that can be used to rotate
an image containing a streak or multiple streaks such that the streak is horizontal, or parllel with
the x-axis.
"""
import logging

import numpy as np
import cv2
import gaussian as gs


def get_edge_intersections(rho, theta, image_dim, scikit_cart_coord):
    """Finds Cartesian Coordinates of a line at the edge of image given the radius and angle
    in Polar Coordinates.

    Parameters
    -----------
    rho : `float`
        Radius of detected line in Polar Coordinates.
    theta : `float`
        Angle of detected line in Polar Coordinates.
    image_dim: `tuple`
        1x2 tuple with the dimensions of the image being analyzed.
    scikit_cart_coord: `tuple`
        (x,y) cartesian coordinate pair the lies on the hough line being evaluated.

     Returns
    ----------
    p1 : `list`
        Pair of `(x1, y1)` pixel coordinates where line enters image.
    p2 : `list`
        Pair of `(x2, y2)` pixel coordinates where line exits image.
    """

    x_size = image_dim[1]
    y_size = image_dim[0]

    # determine where line would be if it were to hit every edge of the image
    if theta % np.pi == 0:
        y = scikit_cart_coord[1]
        one = [0, y]
        two = [x_size, y]
        final_coord = [one, two]
    elif np.cos(theta) == 0:
        x = scikit_cart_coord[0]
        one = [x, 0]
        two = [x, y_size]
        final_coord = [one, two]
    else:
        one = [0, rho / np.sin(theta)]
        two = [rho / np.cos(theta), 0]
        three = [x_size, -1 * (np.cos(theta) / np.sin(theta) * x_size) + (rho / np.sin(theta))]
        four = [(-1 * (np.sin(theta) / np.cos(theta)) * y_size) + (rho / np.cos(theta)), y_size]

        # isolate the x values and the y values
        validate = [one, two, three, four]
        validate = np.array(validate)
        val_x = validate[:, 0]
        val_y = validate[:, 1]

        # determine if the x-values or y-values given are in the boundaries of the image
        bool_x = [0 <= val_x[i] <= x_size for i in range(4)]
        bool_y = [0 <= val_y[i] <= y_size for i in range(4)]
        true_bool = [bool_x[i] & bool_y[i] for i in range(4)]
        final_coord = validate[true_bool]
    return final_coord


def determine_rotation_angle(coor_1, coor_2):
    """Finds angle at which to rotate the image

    Parameters
    -----------
    coor_set1 : `tuple`
        Any set of cartesian coordinates that lie on the line at which the image is to be rotated
    coor_set2 : `tuple'
        Any set of cartesian coordinatets unequal to the first set of coordinates that lie on
        the line at which the image is to be rotated

    Returns
    --------
    angle : `float`
        Angle, in degrees, at which the image should be rotated so the streak of
        interest will be parallel with the x-axis
    """

    # trigonometry
    if coor_1[0] - coor_2[0] == 0:
        angle = np.pi / 2
    else:
        slope = (coor_1[1] - coor_2[1]) / (coor_1[0] - coor_2[0])
        angle = np.arctan(slope)

    # converting angle from radians to degrees
    angle = (angle * 180 / np.pi)
    # angle = np.linspace(angle - 0.5, angle + 0.5, num=11)
    return angle


def norm_rsmd_test(image):
    """Fits function for any input image.

    Parameters
    -----------
    image : `numpy.array`
        2d array containing a rotated image with an isolated streaak.

    Returns
    --------
    nr2 : `float`
        A float that represents the normalized root squared mean deviation for
        a rotated streak's profile.
    a : `float`
        Amplitude of the streak's profile.
    mu : `float`
        Location where streak's profile peaks.
    width : `float`
        The streak's width.
    """
    x = np.arange(0, image.shape[0], 1)
    y = list(np.median(image, axis=1))
    try:
        a, mu, width = gs.fit(x, y)
    except RuntimeError:
        a, mu, width = False, False, False

    if width is not False:
        yhat = gs.gauss(x, a, mu, width)
        nr2 = gs.nrmsd(x, y, yhat)
    else:
        return False, a, mu, width

    return nr2, a, mu, width


def rotate_image(image, angle, coordinates):
    """Rotates an image containing a streak about that streak's midpoint and determined
    angle of rotation and crops that image for further analysis.

    Parameters
    -----------
    image : `numpy.array`
        Image containing the streaks of interest.
    angle : `float`
        Angle at which a particular streak is to be rotated such that the streak is parallel with
        the x-axis of the image.
    coordinates : `list`
        A list of length two with the entrance and exit cartesian coordinates
        of the streak of interest.

    Returns
    --------
    rotated_image : `numpy.array`
        Image containing the streak of interest rotated such that it is parallel with
        the x-axis and cropped to reduce noise.
    """
    # finding midpoint of line to find point of rotation
    # because pixels have to be integers, this midpoint will be an estimate

    # rotating original image without crop
    rotation_x = (coordinates[1][0] + coordinates[0][0]) // 2
    rotation_y = (coordinates[1][1] + coordinates[0][1]) // 2

    matrix = cv2.getRotationMatrix2D((rotation_x, rotation_y), angle, 1.0)
    rotated_image = cv2.warpAffine(image.astype(float), matrix, (image.shape[1], image.shape[0]))

    # cropping image
    distance = np.sqrt((coordinates[1][0] - coordinates[0][0])**2 +
                       (coordinates[1][1] - coordinates[0][1])**2)

    # it's fine if end is bigger than the array, but if start
    # goes negative we are indexing like array[bigger, lower]
    # and get an empty array
    start, end = int(rotation_y - 50), int(rotation_y + 50)
    if start < 0:
        start = 0

    if distance < image.shape[1]:
        rotated_image = rotated_image[start: end, 0: int(distance)]
    else:
        rotated_image = rotated_image[start: end]

    return rotated_image


def transform_rho_theta(clustered_lines, image, cart_coord):
    """Transforms the polar coordinates of any cluster to the mean edge cartesian
    coordinates.

    Parameters
    -----------
    clustered_lines : `numpy.array`
        Array of hough lines with three columns representing rho, theta, and cluster label.
    image : `numpy.array`
        Image containing the streaks of interest.
    cart_coord : `numpy.array`
        A list of tuples representing x,y coordinattes of points on the hough lines.

    Returns
    --------
    transform_coords : `numpy.array`
        Array of mean (x1, y1) and mean (x2, y2) coordinates and the cluster label.
    """
    dim_x, dim_y = clustered_lines.shape
    # R and theta --> become x1, y2, x2, y2
    clustered_line_coords = np.zeros((dim_x, dim_y+2))
    for i, line in enumerate(clustered_lines):
        (x1, y1), (x2, y2) = get_edge_intersections(line[0], line[1], image.shape, cart_coord[i])
        clustered_line_coords[i, 0] = x1
        clustered_line_coords[i, 1] = y1
        clustered_line_coords[i, 2] = x2
        clustered_line_coords[i, 3] = y2
        clustered_line_coords[i, 4] = line[-1]

    transform_coords = []
    ncluster = int(max(clustered_lines[:, 2])) + 1

    for i in range(ncluster):
        cur_cluster = clustered_line_coords[clustered_line_coords[:, -1] == i][:, 0:-1]
        x1_mean = np.mean(cur_cluster[:, 0])
        y1_mean = np.mean(cur_cluster[:, 1])
        x2_mean = np.mean(cur_cluster[:, 2])
        y2_mean = np.mean(cur_cluster[:, 3])

        transform_coords.append([(x1_mean, y1_mean), (x2_mean, y2_mean)])

    return transform_coords


def complete_rotate_image(clustered_lines, angles, image, cart_coord):
    """Creates a rotated image for each cluster of lines if the cluster
    passes validation checks.
    Parameters
    -----------
    clustered_lines : `array`
        Array of lines with each row corresponding to a single houghline,
        the first column corresponding to rho, the second column corresponding to theta,
        and the third column corresponding to the cluster.
    angle : `array`
        Array of angles determined by sckit image for each houghline.
    image : `numpy.array`
        A 2D array containing the image with the streaks of interest.

    Return
    --------
    rotated_imaages : `list`
        List of 2D matrices, each `numpy.array` represents a rotated image of a single cluster.
    best_fit_params: `dictionary`
        Best fit parameters that corresponds to each rotated image that passes validation checks.
    """

    clustered_line_mean_coords = transform_rho_theta(clustered_lines, image, cart_coord)
    rotated_images = []
    best_fit_params = []
    counter = 0
    for line_mean_coord in clustered_line_mean_coords:
        counter += 1
        mean_angle = determine_rotation_angle(line_mean_coord[0], line_mean_coord[1])
        rotated_image = rotate_image(image.astype(float), mean_angle, line_mean_coord)
        is_streak_okay, a, mu, sigma, fwhm = gs.fit_image(rotated_image)
        logging.info(f"Streak okay? = {is_streak_okay}")

        if is_streak_okay:
            best_rotated_image = rotated_image
            angles = np.arange(mean_angle - 0.5, mean_angle + 0.5, 0.1)
            nrsmd_min = np.inf
            for alpha in angles:
                nrsmd_test, a_t, mu_t, width_t = norm_rsmd_test(rotated_image)
                if nrsmd_test is not False and nrsmd_test < nrsmd_min:
                    if not np.isinf(nrsmd_min):
                        a, mu, sigma, fwhm = a_t, mu_t, width_t, width_t * 2.355
                    nrsmd_min = nrsmd_test
                    best_rotated_image = rotated_image
            rotated_images.append(best_rotated_image)
            best_fit_params.append({'amplitude': a,
                                    'mean_brightness': mu,
                                    'sigma': sigma,
                                    'fwhm': fwhm})

    return rotated_images, best_fit_params
