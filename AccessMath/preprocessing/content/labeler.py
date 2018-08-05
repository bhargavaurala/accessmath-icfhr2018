
#==================================================
#  Class that defines operations related to
#  Labeling of whiteboard content from videos
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     June 2015
#
#==================================================

import ctypes

import numpy as np
import scipy.ndimage.measurements as sci_mes

from AM_CommonTools.data.connected_component import ConnectedComponent


class Labeler:
    MIN_CC_PIXELS = 20

    accessmath_lib = ctypes.CDLL('./accessmath_lib.so')

    #========================================================================
    # Label and extract connected components from a binary image
    # (original function)
    #========================================================================
    @staticmethod
    def extractConnectedComponents(content, filter_small=True):
        #start by labeling them....
        labels, count_labels = sci_mes.label( content )

        #get their boundaries and counts
        mins_y = []
        mins_x = []
        maxs_y = []
        maxs_x = []
        counts = []
        for i in range(count_labels):
            mins_y.append( content.shape[0] )
            mins_x.append( content.shape[1] )
            maxs_y.append( 0 )
            maxs_x.append( 0 )
            counts.append( 0 )

        for y in range( content.shape[0] ):
            for x in range( content.shape[1] ):
                if labels[y, x] > 0:
                    #belongs to certain cc
                    cc_id = labels[y, x] - 1

                    #check for change in boundaries
                    if mins_y[cc_id] > y:
                        mins_y[cc_id] = y

                    if maxs_y[cc_id] < y:
                        maxs_y[cc_id] = y

                    if mins_x[cc_id] > x:
                        mins_x[cc_id] = x

                    if maxs_x[cc_id] < x:
                        maxs_x[cc_id] = x

                    #increment count
                    counts[cc_id] += 1

        #finally extract the binary image of each and create the object
        components = []
        for cc_id in range(count_labels):
            #check size
            #...less than thresshold is considered noise...
            if not filter_small or counts[cc_id] >= Labeler.MIN_CC_PIXELS:
                #copy limits (for efficiency)
                max_x = maxs_x[cc_id]
                max_y = maxs_y[cc_id]
                min_x = mins_x[cc_id]
                min_y = mins_y[cc_id]

                height = max_y - min_y + 1
                width = max_x - min_x + 1

                cc_img = np.zeros( (height, width) )

                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if labels[y, x] == cc_id + 1:
                            cc_img[y - min_y, x - min_x] = 255.0

                component = ConnectedComponent(cc_id, min_x, max_x, min_y, max_y, counts[cc_id], cc_img)

                components.append( component )

                #un-comment to output the cc of the sketch...
                #cv2.imwrite( 'out_cc//cc_' + str(self.id) + '_' + str(cc_id) + '.bmp', cc_img)


        return components

    #========================================================================
    # Label and extract connected components from a binary image
    #========================================================================
    @staticmethod
    def extractSpatioTemporalContent(content, ages, filter_small=True):
        # should be a binary image (single channel)
        assert len(content.shape) == 2

        height = content.shape[0]
        width = content.shape[1]

        #start by labeling them....
        labels, count_labels = sci_mes.label( content )

        if count_labels == 0:
            # no CC?
            return []

        # input
        labels_p = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        ages_p = ages.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # output arrays ...
        out_mins_y = np.zeros(count_labels, dtype=np.int32)
        out_maxs_y = np.zeros(count_labels, dtype=np.int32)
        out_mins_x = np.zeros(count_labels, dtype=np.int32)
        out_maxs_x = np.zeros(count_labels, dtype=np.int32)
        out_counts = np.zeros(count_labels, dtype=np.int32)
        out_ages = np.zeros(count_labels, dtype=np.float32)

        # output pointers ....
        out_mins_y_p = out_mins_y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_maxs_y_p = out_maxs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_mins_x_p = out_mins_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_maxs_x_p = out_maxs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        out_counts_p = out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_ages_p = out_ages.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # call C function that finds boundaries, pixel count and minimum age for each CC.
        arg_types = [
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        ]
        Labeler.accessmath_lib.CC_AgeBoundaries.argtypes = arg_types
        Labeler.accessmath_lib.CC_AgeBoundaries.restype = ctypes.c_int32
        Labeler.accessmath_lib.CC_AgeBoundaries(labels_p, ages_p, width, height, count_labels,
                                               out_mins_y_p, out_maxs_y_p, out_mins_x_p, out_maxs_x_p,
                                               out_counts_p, out_ages_p)

        components = []
        for cc_id in range(count_labels):
            # original = (labels == cc_id + 1).sum()
            # print("CC " + str(cc_id) + ", original count: " + str(original) + ", count:" + str(out_counts[cc_id]))

            # check size
            # ...less than thresshold is considered noise...
            if not filter_small or out_counts[cc_id] >= Labeler.MIN_CC_PIXELS:
                max_x = out_maxs_x[cc_id]
                max_y = out_maxs_y[cc_id]
                min_x = out_mins_x[cc_id]
                min_y = out_mins_y[cc_id]

                cc_img = (labels[min_y:max_y+1, min_x:max_x + 1] == cc_id + 1).astype(np.uint8) * 255

                component = ConnectedComponent(cc_id, min_x, max_x, min_y, max_y, out_counts[cc_id], cc_img)
                component.start_time = out_ages[cc_id]
                component.end_time = out_ages[cc_id]

                components.append(component)

        return components