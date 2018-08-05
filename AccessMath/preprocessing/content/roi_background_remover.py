
import cv2
import numpy as np
import scipy.ndimage.measurements as ms

class ROIBackgroundRemover:
    @staticmethod
    def remove_background(binary_images, ROI_polygon, min_overlap, verbose=False, save_prefix=None):
        compressed_output = []

        #background_mask = np.ones(ROI_polygon.shape, ROI_polygon.dtype) * 255
        #background_mask = (ROI_polygon == 0)

        for idx, image in enumerate(binary_images):
            if verbose:
                print("Processed: " + str(idx) + " of " + str(len(binary_images)), end="\r")

            # get the connected components from the original image
            cc_labels, count_labels = ms.label(image)

            # Add the values of pixels on the binary ROI mask per CC
            label_list = range(count_labels + 1)
            size_sums = ms.sum(image, cc_labels, label_list)
            overlap_sums = ms.sum(ROI_polygon, cc_labels, label_list)

            size_sums[0] = 1 # avoid division by zero warning

            # compute the overlap percentage between each CC and the ROI mask
            prop_sums = overlap_sums / size_sums

            # will delete any CC that does not overlap enough with the ROI mask
            to_delete = prop_sums < min_overlap

            delete_mask = to_delete[cc_labels]
            image[delete_mask] = 0

            # erase background
            #image[background_mask] = 0

            # add to buffer
            flag, output_image = cv2.imencode(".png", image)
            compressed_output.append(output_image)

            # debug, save to disk
            if save_prefix is not None:
                cv2.imwrite(save_prefix + "_content_" + str(idx) + ".png", image)

        if verbose:
            print("")

        return compressed_output