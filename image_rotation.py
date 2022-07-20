'''
Author: Kilando Chambers

This module contains the image rotation class that can be used to rotate the
an image containing a streak such that the streak is horizontal, or parllel with
the x-axis.  Use the line_detection_testing.ipynb file to import this module and 
apply it on multiple images.

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

Next steps from here:
    Plot a histogram of the brightness of the image
'''

#Importing necessary libraries
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Create class with inheritance #
class ImageRotation:
    def __init__(self):
        #considering if self.image and self.polar_coor should be required initialized variables
        self.image = None
        self.angle = None
        self.coordinates = None
        self.polar_coor = None
        self.rotated_image = None
        self.mean = None
        self.median = None
 

    def get_coord(self, rho, theta):
        '''Finds Cartesian Coordinates of a line at the edge of image given the radius and angle
        in Polar Coordinates
        
        Parameters
        __________
        rho : 'float'
        Radius of detected line in Polar Coordinates
        
        theta : 'float'
        Angle of detected line in Polar Coordinates
        '''

        x_size = self.image.shape[1]
        y_size = self.image.shape[0]

        #first guess coordinates if they are within the boundaries of the image
        x1, y1 = (0, rho / np.sin(theta))
        x2, y2 = (rho / np.cos(theta), 0)

        slope = y1 / x2

        #adjust coordinates to certainly fit within the boundaries of the image
        if x2 > x_size:
            x2 = x_size
            y2 = (-1 * (np.cos(theta) / np.sin(theta)) * x2) + (rho / np.sin(theta))
            #formula can be found on opencv2 Hough Transform tutorial
        
        if y1 > y_size:
            y1 = y_size
            x1 = (-1 * (np.sin(theta) / np.cos(theta)) * y1) + (rho / np.cos(theta))
        
        return [x1, y1], [x2, y2]

    def coord_all_lines(self, polar_coor):
        #takes list of (rho, theta) coordinates and returns a list of converted cartesian coordinates
        self.coordinates = [self.get_coord(line[0][0],line[0][1]) for line in polar_coor]
        lines_coords = self.coordinates
        return lines_coords

    def summarized_coordinates(self, coordinates, method):
        '''Finds mean edge cartesian coordinates of all Hough Transform lines
        
        Parameters
        __________
        coordinates : 'list'
        List of two pair of cartesian coordinates for each line, where one pair is where
        the streak enters the image and the other is where the streak exits the image 
        '''
        #creates a single list with clusters of x1, y1, x2, y2 values
        points_sep = [points[i][j] for i in range(2) for j in range(2) for points in coordinates]
        #determining the number of elements in each cluster of subcoordinates (e.g. the number of total x1s)
        mod = int(len(points_sep)/4)

        #finding finding mean or median of each subcoordinate to generate entrance and exit coordinates
        if method == 'mean':
            summ_coor = [np.mean(points_sep[i * mod : ((i + 1) * mod)]) for i in range(4)]
            self.mean = ([summ_coor[0], summ_coor[1]], [summ_coor[2], summ_coor[3]])
        elif method == 'median':
            summ_coor = [np.median(points_sep[i * mod : ((i + 1) * mod)]) for i in range(4)]
            self.median = ([summ_coor[0], summ_coor[1]], [summ_coor[2], summ_coor[3]])

        return [summ_coor[0], summ_coor[1]], [summ_coor[2], summ_coor[3]]

    def image_angle(self, coor_1, coor_2):
        '''Finds angle at which to rotate the image
        
        Parameters
        __________
        coor_set1 : 'tuple'
        Any set of cartesian coordinates that lie on the line at which the image is to be rotated
        
        coor_set2 : 'tuple'
        Any set of cartesian coordinatets unequal to the first set of coordinates that lie on the line at
        which the image is to be rotated
        '''

        #trigonometry
        if coor_1[0] - coor_2[0] == 0:
            angle = np.pi / 2

        else: 
            slope = (coor_1[1] - coor_2[1]) / (coor_1[0] - coor_2[0])
            angle = np.arctan(slope)
        
        #converting angle from radians to degrees
        self.angle = angle * 180 / np.pi
        angle_deg = self.angle
        
        return angle_deg

    def rotate_image(self, image, method):

        #finding midpoint of line to find point of rotation
        #because pixels have to be integers, this midpoint will be an estimate      

        #rotating original image without crop

        self.angle = self.image_angle(method[0], method[1])

        method_rotation_x = (method[1][0] + method[0][0]) // 2
        method_rotation_y = (method[1][1] + method[0][1]) // 2

        matrix = cv2.getRotationMatrix2D((method_rotation_x, method_rotation_y), self.angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (self.image.shape[1], self.image.shape[0]))

        #cropping image
        new_height = self.image.shape[0] // 10

        #detecting edge cases where the 10% rule does not apply

        if method_rotation_y - new_height < 0:
            new_height = method_rotation_y
        elif method_rotation_y + new_height > self.image.shape[0]:
            new_height = self.image.shape[0] - method_rotation_y

        rotated_image = rotated_image[int(method_rotation_y - new_height): int(method_rotation_y + new_height)]

        return rotated_image


        
        


        
