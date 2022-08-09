import line_detection_updated as ld
import image_rotation as ir
from astropy.io import fits
import astropy.visualization as aviz
import os

def file_ingest(filepath):
    if not os.path.isfile(filepath):
        raise ValueError("File path is not a file. Expected a fits file.")

    filename = os.path.basename(filepath)
    hdul = fits.open(filepath, cache = True)

    images = []
    images_index = []
    counter = 0

    for i in hdul:
        if i.is_image and i.data is not None:
            if 'EXTTYPE' in i.header:
                if 'IMAGE' in i.header['EXTTYPE']:
                    images.append(i)
                    images_index.append(counter)
            else:
                images.append(i)
                images_index.append(counter)
            
            counter+=1
    
    return {'image_list': images, 'image_indices': images_index, 'filename': filename}

def satmetrics(filepath):
    #Ingest the file
    ingest_dict = file_ingest(filepath)
    images_indices = ingest_dict['image_indices']
    images = ingest_dict['image_list']
    filename = ingest_dict['filename']

    detector = ld.LineDetection() # need configuration from file

    for i in images_indices:
        detector.image = images[i].data.copy()
        results_hough_transform = detector.hough_transformation()
        clustered_lines = ld.cluster(results_hough_transform["Cartesian Coordinates"], results_hough_transform["Lines"])

        rotated_images = ir.rotate_img_clustered(clustered_lines = clustered_lines,
                                                angles = results_hough_transform["Angles"], 
                                                image = results_hough_transform["Thresholded Image"])

        identifier = filename + str(i)

        #Add the final relevant results
        '''
        results_dict = {'file_identifier':identifier,
                        'line_polar_coordinates':,
                        'FWHM':,
                        'sigma':,
                        'amp':,
                        'mean':,
                        'magnitude':}


        
        '''

    #return results_dict