
import cv2
import numpy as np
import scipy.ndimage.measurements as ms

class MLBinarizer:

    def __init__(self, classifier, patch_size=7, sigma_color=13.5, sigma_space=4.0, median_blur_k=33, dark_bg=False):
        self.classifier = classifier
        self.patch_size = patch_size

        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.median_blur_k = median_blur_k
        self.dark_background = dark_bg

    def preprocessing(self, raw_image, get_background=False):
        assert len(raw_image.shape) == 3

        # Apply Bilateral filter to smooth input
        color_filtered = cv2.bilateralFilter(raw_image, -1, self.sigma_color, self.sigma_space)

        blurred_bg = cv2.medianBlur(color_filtered, self.median_blur_k)
        subtracted = color_filtered.astype(np.int32) - blurred_bg.astype(np.int32)

        if self.dark_background:
            # dark background, ignore pixels darker than background (assume part of background)
            subtracted[subtracted < 0] = 0
        else:
            # light background, ignore pixels lighter than background (assume part of background)
            subtracted[subtracted > 0] = 0
            # make the difference positive
            subtracted = np.abs(subtracted)

        subtracted = np.max(subtracted, axis=2)

        if get_background:
            return subtracted, blurred_bg
        else:
            return subtracted

    def preprocessed_binarize(self, preprocessed_input):
        assert len(preprocessed_input.shape) == 2

        h, w = preprocessed_input.shape
        half_patch_size = int((self.patch_size - 1) / 2)
        padded_image = MLBinarizer.image_padding(preprocessed_input, half_patch_size)

        window_pixels = self.patch_size * self.patch_size

        batch_features = np.lib.stride_tricks.as_strided(padded_image,
                                                         shape=[h, w, self.patch_size, self.patch_size],
                                                         strides=padded_image.strides + padded_image.strides)
        batch_features = batch_features.reshape((h * w, window_pixels))

        col_labels = self.classifier.predict(batch_features)

        return col_labels.reshape((h, w)).astype(np.uint8) * 255

    def binarize(self, color_image):
        assert len(color_image.shape) == 3

        preprocessed = self.preprocessing(color_image)

        return self.preprocessed_binarize(preprocessed)

    def hysteresis_binarize(self, color_image):
        preprocessed = self.preprocessing(color_image)

        low_bin = self.preprocessed_binarize(preprocessed)

        otsu_t, high_bin = cv2.threshold(preprocessed.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return MLBinarizer.binary_hysteresis(low_bin, high_bin)

    def binarize_bboxes(self, image, text_boxes, classifier, confidences=None, min_td_confidence=None):
        # first, compute the binarization for the whole image ...
        full_binary = self.hysteresis_binarize(image)

        # second, mask and filter ... only keep the regions of interest
        mask = np.zeros((image.shape[0], image.shape[1]), np.bool)

        # for each bounding box
        for bbox_idx, (x1, y1, x2, y2) in enumerate(text_boxes):
            if confidences is not None and confidences[bbox_idx] < min_td_confidence:
                # Do not use this box!
                continue

            # mark the bounding box area as area of interest
            box_x1 = max(x1, 0)
            box_y1 = max(y1, 0)
            mask[box_y1:y2 + 1, box_x1:x2 + 1] = True

        # final image ...
        binary_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        # ... apply mask ...
        binary_image[mask] = full_binary[mask]

        return binary_image

    def binarize_bboxes_OTSU(self, image, text_boxes, confidences=None, min_td_confidence=None):
        binary_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        edge_image, background = self.preprocessing(image, True)
        edge_image = edge_image.astype(np.uint8)

        for bbox_idx, (x1, y1, x2, y2) in enumerate(text_boxes):
            if confidences is not None and confidences[bbox_idx] < min_td_confidence:
                # Do not use this box!
                continue

            box_x1 = max(x1, 0)
            box_y1 = max(y1, 0)
            text_cut = edge_image[box_y1:y2 + 1, box_x1:x2 + 1]
            bin_cut = binary_image[box_y1:y2 + 1, box_x1:x2 + 1]
            otsu_t, text_cut_bin = cv2.threshold(text_cut, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # only what is white on both images will remain white
            # this means, binarized text will become black on the binary image ...
            binary_image[box_y1:y2 + 1, box_x1:x2 + 1] = np.bitwise_or(bin_cut, text_cut_bin)

        return binary_image


    @staticmethod
    def image_padding(raw_image, n_pixels):
        assert len(raw_image.shape) == 2
        h, w = raw_image.shape

        padded_image = np.zeros((h + n_pixels * 2, w + n_pixels * 2), raw_image.dtype)
        padded_image[n_pixels:n_pixels + h, n_pixels:n_pixels + w] = raw_image.copy()

        # top-left corner
        padded_image[0:n_pixels, 0:n_pixels] = raw_image[n_pixels - 1::-1, n_pixels - 1::-1].copy()
        # top-right corner
        padded_image[0:n_pixels, -n_pixels:] = raw_image[n_pixels - 1::-1, -1:-n_pixels - 1:-1].copy()
        # bottom-left corner
        padded_image[-n_pixels:, 0:n_pixels] = raw_image[-1:-n_pixels - 1:-1, n_pixels - 1::-1].copy()
        # bottom-right corner
        padded_image[-n_pixels:, -n_pixels:] = raw_image[-1:-n_pixels - 1:-1, -1:-n_pixels - 1:-1].copy()

        # left
        padded_image[n_pixels:-n_pixels, 0:n_pixels] = raw_image[:, n_pixels - 1::-1].copy()
        # right
        padded_image[n_pixels:-n_pixels, -n_pixels:] = raw_image[:, -1:-n_pixels - 1:-1].copy()
        # top
        padded_image[0:n_pixels, n_pixels:-n_pixels] = raw_image[n_pixels - 1::-1].copy()
        # bottom
        padded_image[-n_pixels:, n_pixels:-n_pixels] = raw_image[-1:-n_pixels - 1:-1].copy()

        return padded_image


    @staticmethod
    def binary_hysteresis(binary_low, binary_high):
        # get the connected components from the lower threshold image
        low_cc_labels, cc_count_labels = ms.label(binary_low)

        # Add the values of pixels on the binary mask per CC of the binary input
        mask_sums = ms.sum(binary_high, low_cc_labels, range(cc_count_labels + 1))

        # find those CC where the sum is 0 (they do not have a single pixel in the binary mask)
        filtered_labels = mask_sums == 0
        # create a 2D mask of the CC that will be filtered ...
        filter_mask = filtered_labels[low_cc_labels]

        # create a copy of input image ...
        result = binary_low.copy()
        # apply the filter on the copy
        result[filter_mask] = 0

        return result



