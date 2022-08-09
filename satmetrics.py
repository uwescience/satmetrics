import line_detection_updated as ld
import image_rotation as ir
import gaussian
from astropy.io import fits
import astropy.visualization as aviz
import os
import argparse

#Parsing the arguments to import data files
parser = argparse.ArgumentParser()
parser.add_argument('data_files', type=str, help='paths to the data files')

def args2data(parser):
    '''
    Parse the filenames from data_files
    '''
    input_file = parser.data_files

    with open(input_file, 'r') as f:
        files = f.read().splitlines()

    return files

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

    valid_streaks = {}

    for i in images_indices:

        # Performing Hough Transformation
        detector.image = images[i].data.copy()
        results_hough_transform = detector.hough_transformation()
        clustered_lines = ld.cluster(results_hough_transform["Cartesian Coordinates"], results_hough_transform["Lines"])
        subfile_identifier = filename + '-' + str(i)

        #Rotating the image for analysis
        rotated_images = ir.rotate_img_clustered(clustered_lines = clustered_lines,
                                                angles = results_hough_transform["Angles"], 
                                                image = images[i].data,
                                                cart_coord=results_hough_transform['Cartesian Coordinates'])

        valid_streaks_image = {}
        
        #Validating streaks and getting metrics
        for j in range(len(rotated_images)):
            valid, a, mu, sigma, fwhm = gaussian.fit_image(rotated_images[j])
            image_results = {'amplitude':a, 
                            'mean_brightness': mu,
                            'sigma':sigma,
                            'fwhm': fwhm}

            if valid:
                valid_streaks_image['j'] = image_results
        
        valid_streaks[subfile_identifier] = valid_streaks_image
    
    return valid_streaks, images

if __name__ == '__main__':
    
    args = parser.parse_args()
    provided_files = args2data(args)
    print(provided_files)
    
    files = provided_files      
    results = {}
    for filepath in files:
        streak_results, all_images = satmetrics(filepath)
        results[filepath] = streak_results
        print(f"Main file = {filepath}")
        print(f"This file contains {len(all_images)} science images")

        for subfile in streak_results.keys():
            print(f"sub-file name = {subfile}")
            print(f"This image has {len(streak_results[subfile].keys())} valid streaks")

            for streak in streak_results[subfile].keys():
                print(f"streak - {streak}")
                streak_properties = streak_results[subfile][streak]
                print(f"streak amplitude = {streak_properties['amplitude']}")
                print(f"streak mean brightness = {streak_properties['mean_brightness']}")
                print(f"streak width = {streak_properties['sigma']}")
                print(f"streak fwhm = {streak_properties['fwhm']}")

    





