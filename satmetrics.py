import os
import argparse
import logging
import traceback
import sys
import yaml

from astropy.io import fits

import line_detection_updated as ld
import image_rotation as ir


handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    handlers=[handler, ]
    )


def file_ingest(filepath):
    """
    Takes an input fits file and extracts individual science images from it

    Parameters
    ----------
    filepath : `str`
        The path for the input image

    Returns
    --------
    image_list : `list`
        List of all extracted images
    image_indices : `list`
        Indices of extracted images
    filename : `str`
        basename of the filepath
    """

    if not os.path.isfile(filepath):
        raise ValueError("File path is not a file. Expected a fits file.")

    filename = os.path.basename(filepath)
    hdul = fits.open(filepath, cache=True)

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

            counter += 1

    return {'image_list': images, 'image_indices': images_index, 'filename': filename}


def satmetrics(filepath, config={}):
    """
    Applies Hough transformation, performs image rotation and then finds properties of
    detected streaks

    Parameters
    ----------
    filepath : `str`
        The path for the input image
    config : `dict`, optional, default={}
        Yaml file containing the LineDetection class parameters

    Returns
    --------
    valid_streaks : `dict`
        Nested dictionary containing properties of each valid streak
        in each image of the input filepath
    images : `list`
        Science images extracted from the input file
    """

    # Ingest the file
    ingest_dict = file_ingest(filepath)
    images_indices = ingest_dict['image_indices']
    images = ingest_dict['image_list']
    filename = ingest_dict['filename']

    valid_streaks = {}

    for i in images_indices:
        # Performing Hough Transformation
        detector = ld.LineDetection(image=images[i].data)
        if config:
            detector.configure_from_file(config)

        results_ht = detector.hough_transformation()
        if len(results_ht["Lines"]) == 0:
            logging.info(f"No lines found in {filepath} - skipping.")
            return None, None
        clustered_lines = ld.cluster(results_ht["Cartesian Coordinates"],
                                     results_ht["Lines"])
        subfile_identifier = filename + '-' + str(i)

        # Rotating the image for analysis
        rotated_images, best_fit_params = ir.complete_rotate_image(
                                                clustered_lines=clustered_lines,
                                                angles=results_ht["Angles"],
                                                image=images[i].data,
                                                cart_coord=results_ht['Cartesian Coordinates'])

        num_streaks = len(rotated_images)
        valid_streaks_image = {}
        for j in range(num_streaks):
            valid_streaks_image[str(j)] = best_fit_params[j]

        valid_streaks[subfile_identifier] = valid_streaks_image

    return valid_streaks, images


if __name__ == '__main__':
    # Parsing the arguments to import data files
    parser = argparse.ArgumentParser(description="Detect streaks on input images")
    parser.add_argument('fits', nargs='+',
                        help='A text file containing a list of fits file or a list of fits files.')
    parser.add_argument('--config', nargs='?',
                        help="Yaml file containing the LineDetection class parameters.", default={})
    parser.add_argument('--output', nargs='?', help="A yaml file containing the outputs.")
    args = parser.parse_args()

    if len(args.fits) == 1:
        file = args.fits[0]
        filename = os.path.basename(file)
        extension = ".".join(filename.split(".")[1:])

        if 'fits' in extension:
            files = args.fits
        elif os.path.isfile(file):
            with open(file, 'r') as f:
                files = f.readlines()
                files = [x.strip() for x in files]
        else:
            raise ValueError("Expcted a path to fits file or a path to a text file.")

    else:
        files = args.fits

    results = {}
    for filepath in files:
        try:
            streak_results, all_images = satmetrics(filepath, args.config)
        except ValueError as e:
            logging.error(f"Skipping {filepath} due to: {e} ")
            logging.error(traceback.format_exc())
            continue

        # if no lines are found skip further processing
        if streak_results is None:
            continue

        results[filepath] = streak_results
        logging.info(f"Main file = {filepath}")
        logging.info(f"This file contains {len(all_images)} science images")

        for subfile in streak_results.keys():
            logging.info(f"sub-file name = {subfile}")
            logging.info(f"This image has {len(streak_results[subfile].keys())} valid streaks")

            for streak in streak_results[subfile].keys():
                logging.info(f"streak - {streak}")
                streak_properties = streak_results[subfile][streak]
                logging.info(f"streak amplitude = {streak_properties['amplitude']}")
                logging.info(f"streak mean brightness = {streak_properties['mean_brightness']}")
                logging.info(f"streak width = {streak_properties['sigma']}")
                logging.info(f"streak fwhm = {streak_properties['fwhm']}")

    yaml_results = {}
    for filepath in results:
        fname = os.path.basename(filepath)
        yaml_results[fname] = {}
        for subfile in results[filepath]:
            yaml_results[fname][subfile] = {}
            for id, val in results[filepath][subfile].items():
                yaml_results[fname][subfile][id] = {}

                yaml_results[fname][subfile][id]["amplitude"] = float(val["amplitude"])
                yaml_results[fname][subfile][id]["mean_brightness"] = float(val["mean_brightness"])
                yaml_results[fname][subfile][id]["sigma"] = float(val["sigma"])
                yaml_results[fname][subfile][id]["fwhm"] = float(val["fwhm"])

    # Write out the overall output
    if args.output is not None:
        with open(args.output, 'w') as outfile:
            yaml.dump(yaml_results, outfile)
