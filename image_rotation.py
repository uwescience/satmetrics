"""
Author: Kilando Chambers

This module contains multiple functions that can be used to rotate
an image containing a streak or multiple streaks such that the streak is horizontal, or parllel with
the x-axis. Use the line_detection_testing.ipynb file to import this module and
apply it on multiple images.

Process (in development currently):
    1. Choose image
    2. Process the image:
        a. Apply z_score_trim once to reduce the outlier pixel values
        b. Apply z_score_trim on this processed image again
        c. Standardize and normalize the image using cv2
        d. Perform Canny edge detection
        e. Trim the edges of the image (if chosen)
    3. Perform Hough Transformation on the Canny edge image to get streak coordinates
    4. Convert streak polar coordinates into cartesian coordinates for each cluster
    5. Find the mean coordinates at the edges of the image for each cluster of hough lines
    6. Determine the angle of rotation of each cluster of hough lines for the image
       based on the mean coordinates
    7. Rotate the image for each cluster at the determined angle of rotation
"""

import numpy as np
import cv2
import gaussian as gs


def get_coord(rho, theta, image, scikit_cart_coord):
    """Finds Cartesian Coordinates of a line at the edge of image given the radius and angle
    in Polar Coordinates

    Parameters
    -----------
    rho : `float`
        Radius of detected line in Polar Coordinates
    theta : `float`
        Angle of detected line in Polar Coordinates

     Returns
    ----------
    p1 : `list`
        Pair of `(x1, y1)` pixel coordinates where line enters image.
    p2 : `list`
        Pair of `(x2, y2)` pixel coordinates where line exits image.
    """

    x_size = image.shape[1]
    y_size = image.shape[0]

    # determine where line would be if it were to hit every edge of the image
    if theta % np.pi == 0:
        y = scikit_cart_coord[1]
        one = [0, y]
        two = [x_size, y]
        final_coord = one, two
        return final_coord
    elif np.cos(theta) == 0:
        x = scikit_cart_coord[0]
        one = [x, 0]
        two = [x, y_size]
        final_coord = [one, two]
        return one, two
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


def coord_all_lines(polar_coor, image, sckit_cart_coord):
    """Calls get_coord function to create a list of cartesian coordinates converted
    from polar coordinates

    Parameters
    ---------
    polar_coor : `list' or `array`
        List of Polar Coordinates in a cluster
    image : `numpy.array`
        2d array containing the image with the streaks of interest

    Returns
    --------
    coordinates : `list`
        List of [(x1,y1), (x2,y2)] cartesian coordinates
    """

    # takes list of (rho, theta) coordinates and returns a list of converted cartesian coordinates
    coordinates = [get_coord(polar_coor[i][0], polar_coor[i][1], image,
                             sckit_cart_coord[i]) for i in range(len(polar_coor))]
    return coordinates


def summarized_coordinates(coordinates):
    """Finds mean edge cartesian coordinates of all lines in a cluster

    Parameters
    -----------
    coordinates : `list`
        List of two pair of cartesian coordinates for each line, where one pair is where
        the streak enters the image and the other is where the streak exits the image

    Returns
    ---------
    coordinates : `list`
        [(x1,y1), (x2,y2)], which represents the mean entrance and exit points for all lines
        in the cluster
    """

    # creates a single list with clusters of x1, y1, x2, y2 values
    points_sep = [points[i][j] for i in range(2) for j in range(2) for points in coordinates]
    # determining the number of elements in each cluster of subcoordinates
    # (e.g. the number of total x1s)
    mod = int(len(points_sep)/4)

    # finding finding mean or median of each subcoordinate to generate
    # entrance and exit coordinates
    summ_coor = [np.mean(points_sep[i * mod: ((i + 1) * mod)]) for i in range(4)]

    return [summ_coor[0], summ_coor[1]], [summ_coor[2], summ_coor[3]]


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
    angle = np.linspace(angle - 0.5, angle + 0.5, num=11)

    return angle


def norm_rsmd_test(image):
    """Fits function for any input image

    Parameters
    -----------
    image : `numpy.array`
        2d array containing a rotated image with an isolated streaak

    Returns
    --------
    False: `boolean`
        Returns False if the image does not have a fittable gaussian profile
    nr2 : `float`
        A float that represents the normalized root squared mean deviation for
        a rotated streak's profile
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
        return False

    return nr2


def rotate_image(image, angle, coordinates):
    """Rotates an image containing a streak about that streak's midpoint and determined
    angle of rotation and crops that image for further analysis

    Parameters
    -----------
    image : `numpy.array`
        Image containing the streaks of interest
    angle : `float`
        Angle at which a particular streak is to be rotated such that the streak is parallel with
        the x-axis of the image
    coordinates : `list`
        A list of length two with the entrance and exit cartesian coordinates
        of the streak of interest

    Returns
    --------
    rotated_image : `numpy.array`
        Image containing the streak of interest rotated such that it is parallel with
        the x-axis and cropped to reduce noise
    """
    if isinstance(angle, float):
        angle = [angle, ]

    # finding midpoint of line to find point of rotation
    # because pixels have to be integers, this midpoint will be an estimate

    # rotating original image without crop
    for a in angle:
        rotation_x = (coordinates[1][0] + coordinates[0][0]) // 2
        rotation_y = (coordinates[1][1] + coordinates[0][1]) // 2

        matrix = cv2.getRotationMatrix2D((rotation_x, rotation_y), a, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        # cropping image
        distance = np.sqrt((coordinates[1][0] - coordinates[0][0])**2 +
                        (coordinates[1][1] - coordinates[0][1])**2)
        if distance < image.shape[1]:
            rotated_image = rotated_image[int(rotation_y - 50): int(rotation_y + 50), 0: int(distance)]
        else:
            rotated_image = rotated_image[int(rotation_y - 50): int(rotation_y + 50)]

    return rotated_image


def calc_clustered_angles(clustered_lines, angles):
    ncluster = int(max(clustered_lines[:, 2])) + 1
    angles = np.array(angles)
    rotation_angles = []
    for i in range(ncluster):
        cur_angle = angles[clustered_lines[:, 2] == i]
        rotation_angles.append(np.mean(cur_angle))
    return rotation_angles



def rotate_img_clustered(clustered_lines, angles, image, cart_coord):
    """Creates a rotated image for each cluster of lines

    Parameters
    -----------
    clustered_lines : `array`
        Array of lines with each row corresponding to a single houghline,
        the first column corresponding to rho, the second column corresponding to theta,
        and the third column corresponding to the cluster
    angle : `array`
        Array of angles determined by sckit image for each houghline
    image : `numpy.array`
        A 2D array containing the image with the streaks of interest

    Return
    --------
    rot_images : `list`
        List of 2D matrices, each `numpy.array` represents a rotated image of a single cluster
    """

    rot_images = []
    ncluster = int(max(clustered_lines[:, 2])) + 1

    clustered_lines = np.array(clustered_lines)
    angles = np.array(angles)
    cart_coord = np.array(cart_coord)

    for i in range(ncluster):
        cur_cluster = clustered_lines[clustered_lines[:, 2] == i][:, 0:2]

        if len(cur_cluster) == 1:
            nrsmd_true = np.inf
            cur_angle = angles[clustered_lines[:, 2] == i]
            cur_coord = cart_coord[clustered_lines[:, 2] == i]
            cur_angle = (cur_angle * 180 / np.pi) - 90
            cur_angle = cur_angle[0]
  
            coord = coord_all_lines(polar_coor=cur_cluster, image=image,
                                    sckit_cart_coord=cur_coord)
            rotated_image = rotate_image(image, angle)

            for ang in cur_angle:
                rotated_image = rotate_image(image, angle=ang, coordinates=coord[0])
                nrsmd_test = norm_rsmd_test(rotated_image)
                if nrsmd_test is not False and nrsmd_test < nrsmd_true:
                    nrsmd_true = nrsmd_test
                    best_rotated_image = rotated_image
            if nrsmd_true == np.inf:
                best_rotated_image = rotate_image(image,
                                                  angle=cur_angle[5],
                                                  coordinates=coord[0])
            rot_images.append(best_rotated_image)

        else:
            nrsmd_true = np.inf
            cur_coord = cart_coord[clustered_lines[:, 2] == i]
            list_coord = coord_all_lines(cur_cluster, image, cur_coord)
            coord = summarized_coordinates(list_coord)
            cur_angle = determine_rotation_angle(coord[0], coord[1])

            for ang in cur_angle:
                rotated_image = rotate_image(image, angle=ang, coordinates=coord)
                nrsmd_test = norm_rsmd_test(rotated_image)
                if nrsmd_test is not False and nrsmd_test < nrsmd_true:
                    nrsmd_true = nrsmd_test
                    best_rotated_image = rotated_image
            if nrsmd_true == np.inf:
                best_rotated_image = rotate_image(image,
                                                  angle=cur_angle[5],
                                                  coordinates=coord)
            rot_images.append(best_rotated_image)

    return rot_images

    # rot_images = []
    # ncluster = int(max(clustered_lines[:, 2])) + 1

    # clustered_lines = np.array(clustered_lines)
    # angles = np.array(angles)
    # cart_coord = np.array(cart_coord)

    # for i in range(ncluster):
    #     cur_cluster = clustered_lines[clustered_lines[:, 2] == i][:, 0:2]

    #     if len(cur_cluster) == 1:
    #         nrsmd_true = np.inf
    #         cur_angle = angles[clustered_lines[:, 2] == i]
    #         cur_coord = cart_coord[clustered_lines[:, 2] == i]
    #         cur_angle = (cur_angle * 180 / np.pi) - 90
    #         cur_angle = cur_angle[0]
    #         cur_angle = np.linspace(cur_angle - 0.5, cur_angle + 0.5, num=11)
    #         coord = coord_all_lines(polar_coor=cur_cluster, image=image,
    #                                 sckit_cart_coord=cur_coord)

    #         for ang in cur_angle:
    #             rotated_image = rotate_image(image, angle=ang, coordinates=coord[0])
    #             nrsmd_test = norm_rsmd_test(rotated_image)
    #             if nrsmd_test is not False and nrsmd_test < nrsmd_true:
    #                 nrsmd_true = nrsmd_test
    #                 best_rotated_image = rotated_image
    #         if nrsmd_true == np.inf:
    #             best_rotated_image = rotate_image(image,
    #                                               angle=cur_angle[5],
    #                                               coordinates=coord[0])
    #         rot_images.append(best_rotated_image)

    #     else:
    #         nrsmd_true = np.inf
    #         cur_coord = cart_coord[clustered_lines[:, 2] == i]
    #         list_coord = coord_all_lines(cur_cluster, image, cur_coord)
    #         coord = summarized_coordinates(list_coord)
    #         cur_angle = determine_rotation_angle(coord[0], coord[1])

    #         for ang in cur_angle:
    #             rotated_image = rotate_image(image, angle=ang, coordinates=coord)
    #             nrsmd_test = norm_rsmd_test(rotated_image)
    #             if nrsmd_test is not False and nrsmd_test < nrsmd_true:
    #                 nrsmd_true = nrsmd_test
    #                 best_rotated_image = rotated_image
    #         if nrsmd_true == np.inf:
    #             best_rotated_image = rotate_image(image,
    #                                               angle=cur_angle[5],
    #                                               coordinates=coord)
    #         rot_images.append(best_rotated_image)

    # return rot_images