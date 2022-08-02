'''
Author: Abhilash Biswas

This module creates the line detection class that can be used to isolate the 
polar coordinates of streaks in images. This module only contains the class. Use
the line_detection_testing.ipynb file to import this module and apply it on 
multiple images

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

Next steps from here:
    Take the coordinates of the streak and use that to rotate the streak in the original
    image
    
'''

#Importing necessary libraries
from concurrent.futures import process
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
from sklearn.cluster import MeanShift

import matplotlib.pyplot as plt
from matplotlib import cm


#Defining functions

def image_mask(img,percent):
    #Masks the edges of the image

    mask = np.zeros(img.shape, dtype = int)
        
    x_dim_top = int(img.shape[0]*(percent/2))
    x_dim_bottom = int(img.shape[0]*(1-percent/2))
    y_dim_left = int(img.shape[1]*(percent/2))
    y_dim_right = int(img.shape[1]*(1-percent/2))
    
    return x_dim_top,x_dim_bottom,y_dim_left,y_dim_right


  
#Create class
class LineDetection:
    '''
    Packaging the multiple functions used in line detection into 1 class.
    '''
    
    #Instatiating constructors
    def __init__(self):
        
        '''
        image = the image you want to detect streaks in
        
        mask = Takes True/False. If you want to mask the edges of the image or not
        mask_percent = percentage of edge you want to mask
        
        nstd1_cut = Reduce outlier pixel intensities beyond these many standard
        deviations in the first cut of processing the image
        nstd2_binary_cut = Binary thresholding of the image, after the image has been fit to a 0-255 range
        
        
        threshold = the percentage of diagonal length you want to successfully vote 
                    a line of pixels as a straight line streak
        
        '''
        self.image = None
        
        #Image processing parameters
        self.mask = False
        self.mask_percent = 0.2
        self.nstd1_cut = 3
        self.nstd2_binary_cut = 1
        
        #Line detection parameters
        self.threshold = 0.2

        #Blurring parameters
        self.flux_prop_thresholds = [0.1,0.2,0.3,1]
        self.blur_kernel_sizes = [3,5,9,11]


    def process_image(self):
        #Processes the image (see module description for more details)

        trimmed_image = self.image
        
        # Making first cuts in the image
        # Outliers on the positive side take the value of the cut. Outliers on the negative side take the value 0
        up_limit = trimmed_image.mean() + self.nstd1_cut*trimmed_image.std()
        low_limit = trimmed_image.mean() - self.nstd1_cut*trimmed_image.std()
        trimmed_image[trimmed_image > up_limit] = up_limit
        trimmed_image[trimmed_image <= low_limit] = 0

        #Standardizing the image and moving it away from 0
        processed_image = (trimmed_image-trimmed_image.mean())/trimmed_image.std()
        processed_image -= processed_image.min()

        #Fitting the image to a 0-255 range
        processed_image = (255*(processed_image - processed_image.min()) )/ (processed_image.max() - processed_image.min())

        #Thresholding the processed image 
        limit = processed_image.mean() + self.nstd2_binary_cut*processed_image.std()
        threshold, thresholded_image = cv2.threshold(processed_image, limit, 255, cv2.THRESH_BINARY)
        thresholded_image = cv2.convertScaleAbs(thresholded_image)

        #Masking image to remove any borders
        if self.mask == True:
            x_dim_top,x_dim_bottom,y_dim_left,y_dim_right = image_mask(thresholded_image,self.mask_percent)

            thresholded_image = thresholded_image[x_dim_top:x_dim_bottom,y_dim_left:y_dim_right]

        #Deciding Kernel size of blur based on amount of noise and then blurring the image
        num_bright_pixels = np.sum(thresholded_image > np.mean(thresholded_image))
        x = thresholded_image.shape[0]
        y = thresholded_image.shape[1]
        prop_bright_pixels = num_bright_pixels/(x*y)

        for i in range(len(self.flux_prop_thresholds)):
            if prop_bright_pixels < self.flux_prop_thresholds[i]:
                kernel_blur = self.blur_kernel_sizes[i]
                break
        

        #Blur the image
        blurred_image = cv2.medianBlur(thresholded_image, kernel_blur)


        #Performing Canny edge detection
        edges = cv2.Canny(blurred_image, 0, 200)

        #Detecting Contours 
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_img = np.zeros(edges.shape, dtype=np.uint8)

        #Fitting Minimum area rectangles to the contours
        ratio_threshold = 3 #Ratio of length to width
        constant = 0.000001 #To avoid division by 0

        lines = []
        for cnt in contours: 
            rect = cv2.minAreaRect(cnt) # [x, y, w, h, theta]
            box = cv2.boxPoints(rect)
            box = np.asarray(box, dtype=np.int32)
            
            # lines are really long - i.e. one of their sides
            # is much larger than the other. Let's say if
            # one of the sides is threshold times the other one
            # we reject it!
            (x,y), (w, h), theta = rect
            if (w/(h+constant) > ratio_threshold) or (h/(w+constant) > ratio_threshold):
                lines.append(rect)
                cv2.fillPoly(contours_img, [box], (255, 255, 255))



        return (thresholded_image, blurred_image, edges, contours_img)
    

    #Define function to draw straight lines in hough transformation
    
    def hough_transformation(self):
        '''
        RETURNS
        -----------
        limg = lines drawn from the retrived coordinates of the streak on a blank image 
        edges = the image after Canny edge detection
        lines = Estimated coordinates of the streak in the image
        '''
    
        #Processing the image
        thresholded_image, blurred_image, edges, contours_img = self.process_image()
        
        #Performing Hough transformation
        dimx, dimy = self.image.shape
        diagonal = np.sqrt(dimx**2 + dimy**2)
        thresh = int(self.threshold * diagonal)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        
        #Applying Hough lines to retrieve all possible lines
        h, theta, d = hough_line(edges, theta=tested_angles)

        #Retriving the peaks
        accum, angles, dists = hough_line_peaks(h, theta, d, threshold=thresh)
        print(f"Found {len(dists)} lines.")

        #Storing the returns
        lines = np.vstack((dists,angles)).T
        cart_coords_list = []
        angles_list = []

        for i in range(len(angles)):
            (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            cart_coords_list.append((x0,y0))
            angles_list.append(angles[i])


        return lines, angles_list, cart_coords_list, thresholded_image, blurred_image, edges, contours_img


def show(img, ax=None, show=True, title=None, **kwargs):
#Display an image based on a numpy array in matplotlib
    

    # make the plots bigger
    plt.rcParams["figure.figsize"] = (10,10)

    if ax is None:
        fig, ax = plt.subplots()

    if "uint8" == img.dtype:
        ax.imshow(img, vmin=0, vmax=255, **kwargs)
    ax.imshow(img, **kwargs)

    ax.set_title(title, fontsize=22)

    return ax

def cluster(cart_coords, lines, bandwidth=50):

    cart_coords_array = np.array(cart_coords)
    clustering = MeanShift(bandwidth=bandwidth).fit(cart_coords_array)
    labels = (clustering.labels_).reshape((len(clustering.labels_),1))
    clustered_lines = np.hstack((lines,labels))

    for l in set(clustering.labels_):
        mask = clustering.labels_ == l
        x = cart_coords_array[:,0]
        y = cart_coords_array[:,1]
    
        plt.scatter(x[mask], y[mask], label = l)
        plt.legend()
    
    return clustered_lines















