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
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Create class
class LineDetection:
    '''
    Packaging the multiple functions used in line detection into 1 class.
    '''
    
    #Instatiating constructors
    def __init__(self):
        
        '''
        image = the image you want to detect streaks in
        
        trim = Takes True/False. If you want to trim the edges of the image or not
        trim_percent = percentage of edge you want to trim
        
        nstd1_normalize = Reduce outlier pixel intensities beyond these many standard
        deviations in the first cut of processing the image
        nstd2_normalize = Same as above, but for the second cut of image processing
        
        keeppercent = Threshold the image to keep the top x percent of pixel intensity values
        
        threshold = the percentage of diagonal length you want to successfully vote 
                    a line of pixels as a straight line streak
        
        '''
        self.image = None
        
        #Image processing parameters
        self.trim = None
        self.trim_percent = None
        self.nstd1_normalize = None
        self.nstd2_normalize = None
        self.keeppercent = None
        
        #Line detection parameters
        self.threshold = None
    
    def parameters(self):
        print("Please initiate the following parameters:\n")
        print("image = the image you want to detect streaks in \n")
        print("trim = Takes True/False. If you want to trim the edges of the image or not \n")
        print("trim_percent = percentage of edge you want to trim \n")
        print("nstd1_normalize = Reduce outlier pixel intensities beyond these many standard deviations in the first cut of processing the image \n")
        print("nstd2_normalize = Same as above, but for the second cut of image processing \n")
        print("keeppercent = Threshold the image to keep the top x percent of pixel intensity values \n")
        print("threshold = the percentage of diagonal length you want to successfully vote a line of pixels as a straight line streak \n")

    def show(self,img, ax=None, show=True, title=None, **kwargs):
    #Display an image based on a numpy array in matplotlib
        
        print(type(img))
        print(img.shape)
        # make the plots bigger
        plt.rcParams["figure.figsize"] = (10,10)

        if ax is None:
            fig, ax = plt.subplots()

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

    #Define functions to process the image
    def z_score_trim(self,image, nstd, set_to_limits=None):
        #Standardizing the image to remove outlier pixel intensity values
                
        mean = image.mean()
        std = image.std()

        upperz = mean + nstd*std
        lowerz = mean - nstd*std

        if set_to_limits is None:
            upper = upperz
            lower = lowerz
        else:
            upper, lower = set_to_limits

        image[image > upperz] = upper
        image[image < lowerz] = lower

        return image

    def keep_top_percent(self,image, keep_percent):
        #Thresholding the standardized image 
        #(Assumption: Streaks are the brightest objects in the image)

        hist, bars = np.histogram(image, "auto", density=True)
        cdf = np.cumsum(hist*np.diff(bars))
        cutoff = np.where(cdf > 1-keep_percent)[0][0]
        image[image < bars[cutoff]] = 0

        return image

    def image_trim(self,img,percent):
        #Crops the edges of the image
        
        x_dim_left = int(img.shape[0]*(1-percent))
        x_dim_right = int(img.shape[0]*(percent))

        y_dim_top = int(img.shape[1]*(1-percent))
        y_dim_bottom = int(img.shape[1]*percent)
        img = img[x_dim_left:x_dim_right,y_dim_top:y_dim_bottom]

        return img

    def process_image(self):
        #Processes the image (see module description for more details)

        processed_image = self.z_score_trim(self.image, self.nstd1_normalize)
        processed_image = self.z_score_trim(processed_image, self.nstd2_normalize)
        processed_image = self.keep_top_percent(processed_image, self.keeppercent)

        #Standardizing the image and moving it away from 0
        processed_image = (processed_image-processed_image.mean())/processed_image.std()
        processed_image -= processed_image.min()
        processed_image = cv2.normalize(processed_image, processed_image, 0, 255, cv2.NORM_MINMAX)
        processed_image = cv2.convertScaleAbs(processed_image)


        #Creating edges
        threshold, thresholded_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(thresholded_image, 200, 230)
        
        #Cropping image to remove any borders
        if self.trim == True:
            edges = self.image_trim(edges,self.trim_percent)

        return (edges,processed_image, thresholded_image)

    #Define function to draw straight lines in hough transformation

    def draw_lines(self,lines, image, nlines=-1, color=(255, 255, 255), lw=2):
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
        dimx, dimy = image.shape
        # convert to color image so that you can see the lines
        draw_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if isinstance(nlines, slice):
            draw = lines[nlines]
        else:
            draw = lines[:nlines]

        for params in draw:
            if len(params[0]) == 2:
                # this is non-probabilistic hough branch
                rho, theta = params[0]
                x0 = np.cos(theta) * rho
                y0 = np.sin(theta) * rho
                pt1 = (
                    int(x0 - (dimx + dimy) * np.sin(theta)),
                    int(y0 + (dimx + dimy) * np.cos(theta))
                )
                pt2 = (
                    int(x0 + (dimx + dimy) * np.sin(theta)),
                    int(y0 - (dimx + dimy) * np.cos(theta))
                )
                cv2.line(draw_im, pt1, pt2, color, lw)
            else:
                # this is probabilistic hough branch
                cv2.line(draw_im, (params[0, 0], params[0,1]), (params[0, 2], params[0, 3]), color, lw)

        return draw_im
    
    
    def hough_transformation(self):
        '''
        RETURNS
        -----------
        limg = lines drawn from the retrived coordinates of the streak on a blank image 
        edges = the image after Canny edge detection
        lines = Estimated coordinates of the streak in the image
        '''
    
        #Processing the image
        processed_image_tuple = self.process_image()
        
        edges, processed, thresholded_image = processed_image_tuple
        processed_image = processed_image_tuple[1]

        #Performing Hough transformation
        dimx, dimy = processed_image.shape
        diagonal = np.sqrt(dimx**2 + dimy**2)
        threshold = int(self.threshold * diagonal)
        print(f"Threshold: {threshold} pixels")

        lines = cv2.HoughLines(processed_image, 1, np.pi/180, threshold)
        print(f"Found {len(lines)} lines.")

        limg = np.zeros(self.image.shape, dtype=np.uint8)
        limg = self.draw_lines(lines, limg)

        return limg,processed,thresholded_image,edges,lines 

#testing 

