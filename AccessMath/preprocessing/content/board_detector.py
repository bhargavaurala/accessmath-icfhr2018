
import numpy as np
import numpy.numarray.nd_image
import cv, cv2

from helper import Helper
from binarizer import Binarizer

class BoardDetector:
    MethodDefault = 0

    # Detect board detection method
    @staticmethod
    def default_detection(image):
        # get grayscale version
        gray_scale = cv2.cvtColor(image, cv.CV_RGB2GRAY)

        """
        #blurred = cv2.medianBlur(image, 25)

        var_map = Helper.grayscale_variance_map(gray_scale, 5)
        var_map = (var_map / var_map.mean()) * 255

        result = var_map
        """
        # edges ...
        #edges = cv2.Canny(gray_scale, 15, 25, apertureSize = 3)

        #blurred = cv2.medianBlur(image, 25)
        #edges = cv2.Canny(gray_scale, 15, 25, apertureSize = 3)

        """
        square_size = 50
        mean_lum = np.zeros(image.shape)
        for y in range(gray_scale.shape[0] / square_size):
            for x in range(gray_scale.shape[1] / square_size):
                mean_lum[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 0] = image[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 0].mean()
                mean_lum[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 1] = image[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 1].mean()
                mean_lum[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 2] = image[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size, 2].mean()
        """

        result = Binarizer.frameContentBinarization(image, Binarizer.MethodBackgroundSubstraction)
        """
        #create structuring element...
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        #now dilate edges...
        edges = cv2.dilate(edges, strel)

        inverse = 255 - edges

        # label ...
        labels, count_labels = np.numarray.nd_image.label(inverse)

        result = np.zeros(image.shape, np.float32)

        # temporal, use better method to count sizes ...
        largest_size = 0
        largest_mask = 0
        for i in range(1, count_labels + 1):
            mask = (labels == i)

            size = np.count_nonzero(mask)
            if size > largest_size:
                largest_size = size
                largest_mask = i

        mask = (labels == largest_mask)
        result[mask, 0] = image[mask, 0].mean()
        result[mask, 1] = image[mask, 1].mean()
        result[mask, 2] = image[mask, 2].mean()
        """
        return result

    # Detect the board in the given image using the method specified
    @staticmethod
    def detect(image, method_id):
        mask = BoardDetector.default_detection(image)

        return mask

    # Detect and highlights the board in the given image using the method specified
    @staticmethod
    def highlight_board(image, method_id):
        mask = BoardDetector.detect(image, method_id)

        #four corners instead of mask?
        # pending to do the highlighting

        return mask