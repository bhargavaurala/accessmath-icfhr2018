
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import scipy.ndimage.measurements as sci_mes
import numpy as np
import math
import cv2

from AccessMath.util.misc_helper import MiscHelper
from AccessMath.preprocessing.config.parameters import Parameters

class BinaryBackgroundRemover:

    @staticmethod
    def compute_sum(binary_images):
        height, width = binary_images[0].shape

        complete_sum = np.zeros((height, width), dtype=np.float32)

        for image in binary_images:
            complete_sum += (image / 255)

        # normalize the total sum
        max_count = complete_sum.max()
        complete_sum /= max_count

        return complete_sum

    @staticmethod
    def compute_stable_background(complete_sum, high_threshold, low_threshold, min_bg_cc_size, close_radius,
                                  debug_save_prefix=None):

        height, width = complete_sum.shape

        # threshold and obtain low confidence and high confidence background images
        high_image = (complete_sum >= high_threshold).astype(np.uint8) * 255
        low_image = (complete_sum >= low_threshold).astype(np.uint8) * 255

        # Filtering high threshold image
        # first, do a morphological closing on the image
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(close_radius, close_radius))
        high_closed = cv2.morphologyEx(high_image, cv2.MORPH_CLOSE, struct_elem)

        # filtering very small CC in the background image (very likely to be false positives)
        components, count_labels = sci_mes.label(high_closed)
        sizes = sci_mes.sum(high_closed, components, range(count_labels + 1)) / 255.0

        for idx in range(1, count_labels + 1):
            if sizes[idx] < min_bg_cc_size:
                high_closed[components == idx] = 0

        # compute the connected components on the low confidence image ...
        components, count_labels = sci_mes.label(low_image)

        # now delete components from low threshold image that do not overlap the closed version of the high threshold image
        background_model = low_image.copy()
        for idx in range(count_labels):
            cc_mask = components == idx
            if high_closed[cc_mask].sum() == 0:
                # no overlap between current CC and the high confident background image ...
                background_model[cc_mask] = 0

        background_expanded = cv2.morphologyEx(background_model, cv2.MORPH_DILATE, struct_elem)
        background_mask = (background_expanded > 0)

        # Debug
        if debug_save_prefix is not None:
            #tempo_result = np.zeros((height, width, 3), dtype=np.uint8)
            #tempo_result[:,:, 1] = complete_sum * 220 + 32
            tempo_result = complete_sum * 255

            cv2.imwrite(debug_save_prefix + "_sum.png", tempo_result)
            cv2.imwrite(debug_save_prefix + "_low_t.png", low_image)
            cv2.imwrite(debug_save_prefix + "_hi_t.png", high_image)
            cv2.imwrite(debug_save_prefix + "_hi_t_closed.png", high_closed)
            cv2.imwrite(debug_save_prefix + "_model.png", background_model)
            cv2.imwrite(debug_save_prefix + "_model_exp.png", background_expanded)

        return background_mask

    @staticmethod
    def find_otsu_threshold(hist_norm, Q):
        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in xrange(1,255):
            #print(i)
            p1, p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1 = Q[i]
            q2 = Q[255]-Q[i] # cum sum of classes

            b1, b2 = np.hsplit(bins,[i]) # weights

            # finding means and variances
            m1 = np.sum(p1 * b1) / q1
            m2 = np.sum(p2 * b2) / q2

            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        return thresh

    @staticmethod
    def knn_background(complete_sum, bg_mask, content_threshold, noise_threshold, neighbors, verbose=False, debug_save_prefix=None):
        height, width = complete_sum.shape

        # [         min_t         low_t         high_t              ]
        # [ Noise Zone, Content Zone, Neutral Zone, Background Zone ]
        # [ No Class  , Content Class, No Class   , Background Class]

        # Use the original background estimation method as reference for background class
        high_image = bg_mask.astype('uint8') * 255

        low_image = (complete_sum > noise_threshold).astype(np.uint8) * 255
        low_image[high_image > 0] = 0
        low_image[complete_sum >= content_threshold] = 0

        # create training sets ...
        high_count = high_image.sum() / 255
        low_count = low_image.sum() / 255

        bg_pixels = np.transpose(np.nonzero(high_image))
        fg_pixels = np.transpose(np.nonzero(low_image))

        labels = np.zeros(low_count + high_count, dtype=np.int32)
        labels[0:high_count] = 1

        dataset = np.concatenate((bg_pixels, fg_pixels), axis=0)

        """
        high_w = low_count / float(high_count)
        print("Weight for high: " + str(high_w))
        weights = np.ones(low_count + high_count, dtype=np.float32)
        weights[0:high_count] = high_w

        #classifier =  DecisionTreeClassifier("entropy", max_depth=10)
        #classifier = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=4)
        #classifier.fit(dataset, labels, weights)
        """

        if verbose:
            print("Training classifier....")

        classifier = KNeighborsClassifier(neighbors)
        classifier.fit(dataset, labels)

        if verbose:
            print("Classifier trained....")

        # generate visualization
        if verbose:
            print("Generating mask")

        # row by row ...
        background_mask = np.zeros((height, width), np.uint8)
        for row in range(height):
            points = np.zeros((width, 2), dtype=np.float32)
            points[:, 0] = row
            points[:, 1] = np.arange(0, width)

            y = classifier.predict(points)
            background_mask[row, :] = y * 255

        # dilate mask
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(Parameters.BGRem_KNN_dilation, Parameters.BGRem_KNN_dilation))
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_DILATE, struct_elem)

        # invert mask
        inverse_mask = 255 - background_mask
        # label mask
        components, count_labels = sci_mes.label(inverse_mask)
        sizes = sci_mes.sum(inverse_mask, components, range(count_labels + 1)) / 255.0

        # remove CC with less than % of total size
        total_pixels = float(width * height)
        for idx in range(1, count_labels + 1):
            ratio = sizes[idx] / total_pixels

            if ratio < Parameters.BGRem_KNN_min_region_size:
                inverse_mask[components==idx] = 0

        # invert again
        background_mask = 255 - inverse_mask

        # erode mask
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_ERODE, struct_elem)

        background_mask = (background_mask / 255).astype('bool')

        # here must create a classifier ....
        if debug_save_prefix is not None:
            out_mask = np.zeros((height, width, 3), dtype=np.uint8)
            out_mask[:, :, 0] = low_image
            out_mask[:, :, 1] = background_mask.astype('uint8') * 128
            out_mask[:, :, 2] = high_image

            cv2.imwrite(debug_save_prefix + "_knn_mask.png", out_mask)

        return background_mask


    @staticmethod
    def remove_background(binary_images, background_mask, verbose=False, save_prefix=None):
        compressed_output = []
        all_sums = []

        for idx, image in enumerate(binary_images):
            if verbose and ((idx + 1) % 100 == 0):
                print("Processed: " + str(idx) + " of " + str(len(binary_images)))

            # erase background
            image[background_mask] = 0

            current_sum = image.sum() / 255
            all_sums.append(current_sum)

            # add to buffer
            flag, output_image = cv2.imencode(".png", image)
            compressed_output.append(output_image)

            # debug, save to disk
            if save_prefix is not None:
                cv2.imwrite(save_prefix + "_content_" + str(idx) + ".png", image)

        return compressed_output, all_sums