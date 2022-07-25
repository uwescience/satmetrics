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

#Defining functions

def show(img, ax=None, show=True, title=None, **kwargs):
#Display an image based on a numpy array in matplotlib
    
    print(type(img))
    print(img.shape)
    # make the plots bigger
    plt.rcParams["figure.figsize"] = (10,10)

    if ax is None:
        fig, ax = plt.subplots()

    if "uint8" == img.dtype:
        ax.imshow(img, vmin=0, vmax=255, **kwargs)
    ax.imshow(img, **kwargs)

    ax.set_title(title, fontsize=22)

    return ax

def plot_hist(img):
#Plot a histogram
    """Plot a histogram of an image. """
    plt.hist(img.ravel(), bins=255)
    plt.xlabel("Brightness level [0-255]")
    plt.ylabel("Pixel count")
    plt.show()

def image_mask(img,percent):
    #Masks the edges of the image

    mask = np.zeros(img.shape, dtype = int)
        
    x_dim_top = int(img.shape[0]*(percent/2))
    x_dim_bottom = int(img.shape[0]*(1-percent/2))
    y_dim_left = int(img.shape[1]*(percent/2))
    y_dim_right = int(img.shape[1]*(1-percent/2))
    
    return x_dim_top,x_dim_bottom,y_dim_left,y_dim_right

def get_coord(image, rho, theta):
    '''Finds Cartesian Coordinates of a line at the edge of image given the radius and angle
    in Polar Coordinates
    
    Parameters
    __________
    rho : 'float'
    Radius of detected line in Polar Coordinates
    
    theta : 'float'
    Angle of detected line in Polar Coordinates
    '''

    x_size = image.shape[1]
    y_size = image.shape[0]

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

def erode_image(image, kernel):

    kernel = np.ones(kernel, dtype="uint8")
    eroded_image = cv2.erode(image, kernel)

    return eroded_image

def draw_lines(lines, image, nlines=-1, color=(255, 255, 255), lw=2):
    """
    Draws Hough lines on a given image.

    Parameters
    ----------
    lines : `cv2.HoughLines`
        2D array of line parameters, line parameters are stored in [0][x] as
        tuples.
    image : `np.array`
        Image on which to draw the lines
    nlines : `int` or `slice`
        Number of top-voted for lines to draw, or a slice object corectly 
        indexing the lines array. By default draws all lines.
    color : `tuple`
        Tuple of values defining a BGR color.
    lw : `int`
        Line width.
    """

    # Create a blank image to draw on
    draw_im = np.zeros(image.shape, dtype=np.uint8)

    #Get the polar coordinates and convert to cartesian coordinates
    if len(lines) == 1:
        rho = lines[0][0][0]
        theta = lines[0][0][1]

        point_1, point_2 = get_coord(image, rho, theta)
        point_1, point_2 = np.array(point_1, dtype=int),  np.array(point_2, dtype=int)
        try:
            cv2.line(draw_im, point_1.astype("int"), point_2.astype("int"), (255,255,255), 2)
        except:
            print(f"Failed to draw line ({point_1},  {point_2})")        
    

    else:
        for points in lines:
            point_1, point_2 = get_coord(image, points[0][0], points[0][1])
            point_1, point_2 = np.array(point_1, dtype=int),  np.array(point_2, dtype=int)
            try:
                cv2.line(draw_im, point_1.astype("int"), point_2.astype("int"), (255,255,255), 2)
            except:
                print(f"Failed to draw line ({point_1},  {point_2})")        

    
  
    return draw_im
  
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
        self.mask = None
        self.mask_percent = None
        self.nstd1_cut = None
        self.nstd2_binary_cut = None
        self.erode = None
        self.erode_threshold = None
        
        #Line detection parameters
        self.threshold = None


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


        #Blur the image
        blurred_image = cv2.medianBlur(thresholded_image, 3)
        #blurred_image = cv2.convertScaleAbs(blurred_image)


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
        threshold = int(self.threshold * diagonal)
        print(f"Threshold: {threshold} pixels")


        #Performing Hough Transformation on both MAR and Canny image
        lines_mar = cv2.HoughLines(contours_img, 1, np.pi/180, threshold)
        lines_canny = cv2.HoughLines(edges, 1, np.pi/180, threshold)

        print(f"Found {len(lines_mar)} lines on MAR.")
        print(f"Found {len(lines_canny)} lines on Canny.")
        
        if self.erode == True:
            if len(lines_mar) > self.erode_threshold:
                print("Found larger than acceptable number of lines, eroding the image and re-doing Hough Transformation")
                eroded_img = self.erode_image(thresholded_image)

                lines_eroded = cv2.HoughLines(eroded_img, 1, np.pi/180, int(threshold/2))

                print(f"Found {len(lines_eroded)} lines.")

        
        
        limg_mar = np.zeros(self.image.shape, dtype=np.uint8)
        limg_canny = np.zeros(self.image.shape, dtype=np.uint8)
        limg_eroded = np.zeros(self.image.shape, dtype=np.uint8)

        limg_mar = draw_lines(lines_mar, limg_mar)
        limg_canny = draw_lines(lines_canny, limg_canny)

        if self.erode == True:
            limg_eroded = draw_lines(lines_eroded, limg_eroded)

        return limg_mar,limg_canny,limg_eroded, thresholded_image, blurred_image, edges, contours_img, lines_mar, lines_canny 



