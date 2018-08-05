#==================================================
#  Class that defines operations related to
#  binarization of whiteboard content from videos
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     June 2015
#
#==================================================

import ctypes

import cv2
import numpy as np
import scipy.ndimage.measurements

from AccessMath.preprocessing.tools.adaptive_equalizer import AdaptiveEqualizer

class Binarizer:
    DEBUG_COUNT = 0
    accessmath_lib = ctypes.CDLL('./accessmath_lib.so')

    MethodEdgeBased = 1
    MethodBackgroundSubstraction = 2
    MethodChalkboard = 3
    MethodSubtractionK = 4

    #======================================================================
    # Extract the content from a frame using the original method based on
    # edges (2013)
    #======================================================================
    @staticmethod
    def edgeBasedFrameBinarization(frame):
        #assume image is on RGB format,
        #convert to gray scale (8-bits)
        if len(frame.shape) == 3:
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_scale = frame

        #now apply Canny edge detection....
        edges = cv2.Canny(gray_scale, 20, 60, apertureSize = 3)

        #create structuring element...
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        #now dilate edges...
        edges = cv2.dilate(edges, strel)

        #now invert...
        board = cv2.bitwise_not( edges )

        #now label connected components...
        labels, count_labels = scipy.ndimage.measurements.label( board)

        other_labels = labels.copy()

        #board is going to be any region above certain percent...
        percent_board = 0.25
        dim = float(frame.shape[0] * frame.shape[1])
        #sizes = np.numarray.nd_image.sum(board, labels, range(count_labels + 1))
        sizes = scipy.ndimage.measurements.sum(board, labels, range(count_labels + 1))
        remove_size = (sizes / 255.0) < (dim * percent_board)
        remove_pix = remove_size[ labels ]
        labels[ remove_pix ] = 0

        only_board = cv2.compare(labels, 0, cv2.CMP_GT )

        #cv2.imwrite('only_board.bmp', only_board)

        #get the boundaries of the board
        board_y, board_x = np.nonzero( only_board )
        min_board_x = np.min(board_x)
        max_board_x = np.max(board_x)
        min_board_y = np.min(board_y)
        max_board_y = np.max(board_y)

        #keep them as information
        board_box = (min_board_x, max_board_x, min_board_y, max_board_y)

        #try to identify other regions that are inside of the board and are part of the content
        #sometimes, there might be closed regions that end up being considered part of content
        percent_cc = 0.005
        for i in range(1, count_labels):
            if (sizes[i] / 255.0) >= dim * percent_cc and \
               (sizes[i] / 255.0) < dim * percent_board:
                #large enough to be considered....
                #get the box....
                only_component = cv2.compare( other_labels, i, cv2.CMP_EQ)

                cc_y, cc_x = np.nonzero( only_component )

                min_cc_x = np.min(cc_x)
                max_cc_x = np.max(cc_x)
                min_cc_y = np.min(cc_y)
                max_cc_y = np.max(cc_y)

                #if inside of the box of the board, add....
                if min_cc_x > min_board_x and \
                   min_cc_y > min_board_y and \
                   max_cc_x < max_board_x and \
                   max_cc_y < max_board_y:
                    #keep this region as part of board...
                    only_board = cv2.bitwise_or( only_board, only_component )

        #then, the remaining parts are either content or background...
        not_board = cv2.bitwise_not( only_board )

        #label them....
        labels, count_labels = scipy.ndimage.measurements.label(not_board)

        #get the sizes of all CC's that are not the board
        sizes = scipy.ndimage.measurements.sum(not_board, labels, range(count_labels + 1))

        #filter any CC that is above certain % of the total area
        percent = 0.05
        remove_size = (sizes / 255.0) > (not_board.shape[0] * not_board.shape[1] * percent)
        remove_pix = remove_size[ labels ]
        labels[ remove_pix ] = 0

        #everything equal to 0 is background
        only_background = cv2.compare(labels, 0, cv2.CMP_EQ )

        #cv2.imwrite('only_background.bmp', only_background)

        #finally, content is what is not board or background...
        only_content = cv2.bitwise_or(only_board, only_background)

        #cv2.imwrite('only_content.bmp', only_content)
        return board_box, only_content


    #====================================================
    # Separate the content from the background to
    # identify connected components later
    #====================================================
    @staticmethod
    def edgeBasedRegionBinarization(image):
        #works with gray scale image
        if len(image.shape) == 3:
            gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_scale = image

        rows = gray_scale.shape[0]
        cols = gray_scale.shape[1]

        #get Contrast-Limited Adaptive Histogram Equalization
        tiles_y = int(rows / 20) #20
        tiles_x = int(cols / 20) #20

        equalized = AdaptiveEqualizer.adapthisteq(gray_scale, 0.04, tiles_x, tiles_y)

        #now apply Canny edge detection....
        edges = cv2.Canny(gray_scale, 10, 50, apertureSize = 3)

        #dilate the edges...
        #....create structuring element...
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #....now dilate edges...
        dilate = cv2.dilate(edges, strel)

        #now invert...
        board = cv2.bitwise_not( dilate )

        #now label connected components...
        labels, count_labels = scipy.ndimage.measurements.label( board)

        #get the sizes of the connected components...
        sizes = scipy.ndimage.measurements.sum(board, labels, range(count_labels + 1))

        #filter all connected components below certain threshold...
        percent = 0.05
        remove_size = (sizes / 255.0) < (rows * cols * percent)
        remove_pix = remove_size[ labels ]
        labels[ remove_pix ] = 0

        #it has to be above certain percentage to be background
        only_board = cv2.compare(labels, 0, cv2.CMP_GT )

        #...dilate...
        only_board = cv2.dilate( only_board, strel )

        #print("Min " + str(equalized.min()))
        #print("Min " + str(equalized.max()))
        #cv2.imshow("test", equalized / 255.0)
        #cv2.waitKey(0)

        #final_content = np.zeros( equalized.shape )
        final_content = Binarizer.threshold_content(equalized, only_board, 128)

        return final_content


    #======================================================================
    # Second binarization method based on background estimation and
    # subtraction - By Roger S. Gaborski
    #======================================================================
    @staticmethod
    def backgroundSubtractionBinarization(image, bluring_ksize=3, disk_size=14, threshold=0.89, min_pixels=6):
        if len(image.shape) == 3:
            gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_scale = image

        rows = gray_scale.shape[0]
        cols = gray_scale.shape[1]

        # median filtering ....
        blurred = cv2.medianBlur(gray_scale, bluring_ksize)

        # background subtraction
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(disk_size, disk_size))
        img_back = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, struct_elem).astype("float32")

        # remove background
        subtracted = blurred / img_back

        # DISK SIZE = 14
        #cv2.imwrite("output/images/z_sub_" + str(Binarizer.DEBUG_COUNT) + ".png", ((subtracted > 0.8) & (subtracted < 0.93)).astype(np.float32) * 255)
        #Binarizer.DEBUG_COUNT += 1

        # threshold ..
        thresholded = (subtracted > threshold).astype(gray_scale.dtype) * 255

        # invert
        inverted = 255 - thresholded

        # remove CC with less than min_pixels pixels
        # ... label connected components...
        labels, count_labels = scipy.ndimage.measurements.label(inverted)
        # ... get sizes
        sizes = scipy.ndimage.measurements.sum(inverted, labels, range(count_labels + 1)) / 255.0
        # ... sizes to remove ...
        to_remove = sizes < min_pixels
        # ... remove mask ...
        remove_pix = to_remove[labels]
        # ... remove ...
        labels[remove_pix] = 0
        # ... final result ...
        valid_pixels = cv2.compare(labels, 0, cv2.CMP_GT)

        result = valid_pixels

        return result

    #======================================================================
    # Based on background subtraction method but adapted to chalkboards
    #======================================================================
    @staticmethod
    def chalkboardBinarization(image, bluring_ksize=5, disk_size=14, threshold=25, min_pixels=8):
        if len(image.shape) == 3:
            gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_scale = image

        rows, cols = gray_scale.shape

        # median filtering ....
        blurred = cv2.medianBlur(gray_scale, bluring_ksize)

        # background subtraction
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(disk_size, disk_size))
        img_back = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, struct_elem).astype("float32")

        # remove background
        subtracted = gray_scale.astype("float32") - img_back

        # threshold ..
        thresholded = (subtracted > threshold).astype(gray_scale.dtype) * 255

        # remove CC with less than min_pixels pixels
        # ... label connected components...
        labels, count_labels = scipy.ndimage.measurements.label(thresholded)
        # ... get sizes
        sizes = scipy.ndimage.measurements.sum(thresholded, labels, range(count_labels + 1)) / 255.0
        # ... sizes to remove ...
        to_remove = sizes < min_pixels
        # ... remove mask ...
        remove_pix = to_remove[labels]
        # ... remove ...
        labels[remove_pix] = 0
        # ... final result ...
        valid_pixels = cv2.compare(labels, 0, cv2.CMP_GT)

        return subtracted

    #======================================================================
    # Third binarization method based on background estimation and
    # subtraction - By Kenny Davila
    #======================================================================
    @staticmethod
    def bgSubtractionKBinarization(image, dark_background=False, sigmaColor=4.0, sigmaSpace=4.0,  bluring_ksize=51, threshold=7, min_pixels=5):
        if len(image.shape) == 3:
            gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_scale = image

        # Apply Bilateral filter to smooth input
        gray_scale = cv2.bilateralFilter(gray_scale, -1, sigmaColor, sigmaSpace)

        # median filtering for background removal ....
        blurred_bg = cv2.medianBlur(gray_scale, bluring_ksize)

        # remove background
        subtracted = gray_scale.astype(np.int32) - blurred_bg.astype(np.int32)

        if dark_background:
            # dark background, ignore pixels darker than background (assume part of background)
            subtracted[subtracted < 0] = 0
        else:
            # light background, ignore pixels lighter than background (assume part of background)
            subtracted[subtracted > 0] = 0
            # make the difference positive
            subtracted = np.abs(subtracted)

        # threshold ..
        thresholded = (subtracted >= threshold).astype(np.uint8) * 255

        # remove CC with less than min_pixels pixels
        result = Binarizer.filter_small_CC(thresholded, min_pixels)

        return result

    @staticmethod
    def filter_small_CC(binary, min_pixels):
        if min_pixels <= 0:
            return binary

        # ... label connected components...
        labels, count_labels = scipy.ndimage.measurements.label(binary)
        # ... get sizes
        sizes = scipy.ndimage.measurements.sum(binary, labels, range(count_labels + 1)) / 255.0
        # ... sizes to remove ...
        to_remove = sizes < min_pixels
        # ... remove mask ...
        remove_pix = to_remove[labels]
        # ... remove ...
        labels[remove_pix] = 0
        # ... final result ...
        return cv2.compare(labels, 0, cv2.CMP_GT)

    #======================================================================
    # Extract the content from a frame and binarize it using
    # the specified method
    #======================================================================
    @staticmethod
    def frameContentBinarization(frame, method_id):
        if method_id == Binarizer.MethodEdgeBased:
            return Binarizer.edgeBasedFrameBinarization(frame)
        elif method_id == Binarizer.MethodBackgroundSubstraction:
            return Binarizer.backgroundSubtractionBinarization(frame)
        elif method_id == Binarizer.MethodChalkboard:
            return Binarizer.chalkboardBinarization(frame)
        elif method_id == Binarizer.MethodSubtractionK:
            return Binarizer.bgSubtractionKBinarization(frame)

        # invalid method
        return None


    #======================================================================
    # Extract the content from a region (whiteboard content only)
    # using the specified method
    #======================================================================
    @staticmethod
    def regionBinarization(region_image, method_id):
        if method_id == Binarizer.MethodEdgeBased:
            return Binarizer.edgeBasedRegionBinarization(region_image)
        elif method_id == Binarizer.MethodBackgroundSubstraction:
            return Binarizer.backgroundSubtractionBinarization(region_image)
        elif method_id == Binarizer.MethodChalkboard:
            return Binarizer.chalkboardBinarization(region_image)
        elif method_id == Binarizer.MethodSubtractionK:
            return Binarizer.bgSubtractionKBinarization(region_image)

        # invalid method
        return None

    @staticmethod
    def threshold_content(equalized, only_board, threshold):

        height = only_board.shape[0]
        width = only_board.shape[1]

        # create pointers ...
        equalized_p = equalized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        only_board_p = only_board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        result = np.zeros(equalized.shape, dtype=np.uint8)
        result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # call C implementation ...
        arg_types = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32,
                     ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8)]

        Binarizer.accessmath_lib.combine_results.argtypes = arg_types
        Binarizer.accessmath_lib.combine_results.restype = ctypes.c_int32
        Binarizer.accessmath_lib.combine_results(only_board_p, equalized_p, width, height, threshold, result_p)

        return result

