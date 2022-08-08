import line_detection_updated as ld
import image_rotation as ir
from astropy.io import fits
import astropy.visualization as aviz
import os

# push to integrations
def satmetrics(filepath):
    if not os.path.isfile(filepath):
        raise ValueError("File path is not a file. Expected a fits file.")

    filename = os.path.basename(filepath)
    hdul = fits.open(filepath, cache = True)

    images = []
    for i in hdul:
        if i.is_image and i.data is not None:
            images.append(i)
    
    detector = ld.LineDetection() # need configuration from file
    results_dict = {}
    for img_data in images:
        detector.image = img_data.data.copy()
        results = detector.hough_transformation()
        clustered_lines = ld.cluster(results["Cartesian Coordinates"], results["Lines"])
        results["Clustered Lines"] = clustered_lines

        key = filename + str(images.index(img_data))
        results_dict[key] = results
        rotated_images = ir.rotate_img_clustered(clustered_lines = results["Clustered Lines"],
                                                angles = results["Angles"], image = results["Thresholded Image"])

        results_dict[key]["Rotated Images"] = rotated_images
    
    return results_dict