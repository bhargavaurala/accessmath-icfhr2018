

import ctypes

import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AccessMath.util.misc_helper import MiscHelper

class CC_FeatureExtractor:
    def __init__(self, use_C_features, use_aspect_ratio=True, use_stats=True, use_mass_center=True,
                 use_horizontal_crossings=True, use_vertical_crossings=True, use_2D_histogram=True):

        self.useCMode = use_C_features

        if self.useCMode:
            self.accessmath_feats = ctypes.CDLL('./accessmath_features.so')
        else:
            self.accessmath_feats = None

            # different features available
        # 1) Basic features
        self.use_aspect_ratio = use_aspect_ratio
        self.ratio_range = 2.0
        self.use_stats = use_stats
        self.use_mass_center = use_mass_center
        # 2) Crossings
        self.use_horizontal_crossings = use_horizontal_crossings
        self.use_vertical_crossings = use_vertical_crossings
        self.count_crossings = 16  # 8
        # 3) Histogram
        self.use_2D_histogram = use_2D_histogram
        self.histogram_bins = 5
        self.histogram_scale = 10.0

        self.useCMode = True

    def featuresCount(self):
        count = 0
        if self.use_aspect_ratio:
            count += 1

        if self.use_stats:
            count += 5

        if self.use_horizontal_crossings:
            count += self.count_crossings * 3

        if self.use_vertical_crossings:
            count += self.count_crossings * 3

        if self.use_2D_histogram:
            count += self.histogram_bins * self.histogram_bins

        return count


    def getNormalizedAspectRatio(self, cc):
        width = cc.max_x - cc.min_x + 1
        height = cc.max_y - cc.min_y + 1

        #raw aspect ratio....
        aspect_ratio = width / float(height)

        #re-scale and resize
        #0 -> aspect ratio 1.0
        # 0-1 > aspect ratio between 1.0 and ratioRange
        #> 1   > large aspect ratio
        #-1 -0 > aspect ratio between 1.0/ratioRange and 1.0
        # < -1 > aspect ratio below 1.0/ratioRange
        if aspect_ratio >= 1.0:
            n_ratio = (aspect_ratio - 1.0) / self.ratio_range
        else:
            n_ratio = ((-1.0 / aspect_ratio) + 1.0) / self.ratio_range

        return n_ratio

    def getStats(self, cc):
        # calculate center (AVG)
        avg_x = 0
        avg_y = 0

        half_w = cc.normalized.shape[1] / 2.0
        half_h = cc.normalized.shape[0] / 2.0
        count = 0

        for y in range(cc.normalized.shape[0]):
            sy = (y - half_h) / half_h

            for x in range(cc.normalized.shape[1]):
                sx = (x - half_w) / half_w

                if cc.normalized[y, x] > 0.0:
                    avg_x += sx
                    avg_y += sy
                    count += 1

        # center of mass
        avg_x /= count
        avg_y /= count

        # now get the var x, var y and covX-Y
        var_x = 0.0
        var_y = 0.0
        cov_xy = 0.0
        for y in range(cc.normalized.shape[0]):
            sy = (y - half_h) / half_h

            for x in range(cc.normalized.shape[1]):
                sx = (x - half_w) / half_w

                if cc.normalized[y, x] > 0.0:
                    var_x += (sx - avg_x) * (sx - avg_x)
                    var_y += (sy - avg_y) * (sy - avg_y)
                    cov_xy += (sx - avg_x) * (sy - avg_y)

        var_x /= count
        var_y /= count
        cov_xy /= count

        return [ avg_x, avg_y, var_x, var_y, cov_xy ]

    def getCrossings(self, cc, horizontal):
        if horizontal:
            step = cc.normalized.shape[1] / float(self.count_crossings + 1)
        else:
            step = cc.normalized.shape[0] / float(self.count_crossings + 1)

        counts = []
        mins = []
        maxs = []
        for i in range(self.count_crossings):
            pos = int((i + 1) * step)

            # the crossing is threated in terms of boolean intervals...
            booleans = []
            if horizontal:
                #horizontal -> y fixed and x moves
                for x in range(cc.normalized.shape[1]):
                    booleans.append(cc.normalized[pos, x] > 128.0)

            else:
                #vertical -> x fixed and y moves
                for y in range(cc.normalized.shape[0]):
                    booleans.append(cc.normalized[y, pos] > 128.0)

            # find the intervals...
            intervals = MiscHelper.findBooleanIntervals(booleans, True)

            # now, get the middle points for each interval...
            midPoints = MiscHelper.intervalMidPoints(intervals)

            # normalize values
            midPoints = MiscHelper.scaleValues(midPoints, 0, cc.normalized.shape[0] - 1, -1, 1)

            counts.append(len(intervals))

            if len(intervals) > 0:
                mins.append(midPoints[0])
                maxs.append(midPoints[-1])
            else:
                mins.append(1.1)
                maxs.append(-1.1)

        counts = MiscHelper.scaleValues(counts, 0, 10, -3, 3)

        return counts + mins + maxs
        # return counts

    def getHistogram(self, cc):
        #...copy... (makes code shorter)...
        size = ConnectedComponent.NormalizedSize

        counts = np.zeros( (self.histogram_bins, self.histogram_bins) )

        start_y = 0
        for row in range(self.histogram_bins):
            start_x = 0
            height = int( size / self.histogram_bins) + (1 if row < size % self.histogram_bins else 0)

            for col in range(self.histogram_bins):
                #the size of the bin
                width = int( size / self.histogram_bins) + (1 if col < size % self.histogram_bins else 0)

                bin_pixels = cc.normalized[ start_y:(start_y + height), start_x:(start_x + width) ]
                bin_count = np.count_nonzero(bin_pixels)

                counts[row, col] =  bin_count

                start_x += width

            start_y += height

        #now, normalize....
        total = float(counts.sum())

        counts = counts / total

        #now scale ...
        counts = (counts - 0.5) * self.histogram_scale

        return counts.reshape(-1,).tolist()

    def getFeatures(self, cc):
        if self.useCMode:
            #use the faster C implementation instead
            return self.getFeaturesC(cc)

        features = []

        if cc.normalized is None:
            # create a standar resized (and recenter image)
            cc.normalizeImage(ConnectedComponent.NormalizedSize)

        # 1) Basic features
        if self.use_aspect_ratio:
            n_ratio = self.getNormalizedAspectRatio(cc)

            features.append(n_ratio)

        if self.use_stats:
            features += self.getStats(cc)

        # 2) Crossings
        if self.use_horizontal_crossings:
            features += self.getCrossings(cc, True)

        if self.use_vertical_crossings:
            features += self.getCrossings(cc, False)

        # 3) histogram...
        if self.use_2D_histogram:
            features += self.getHistogram()

        mat_features = np.mat(features)

        return mat_features

    def getFeaturesC(self, cc):
        if cc.normalized is None:
            # create a standar resized (and recenter image)
            cc.normalizeImage(ConnectedComponent.NormalizedSize)

        n_features = self.featuresCount()
        mat_features = np.zeros((1, n_features), dtype=np.float64)

        #this is the only feature that can be computed efficiently in python
        #and reduces the work of the C version...
        if self.use_aspect_ratio:
            n_ratio = self.getNormalizedAspectRatio(cc)
            mat_features[0, 0] = n_ratio

        mat_features_p = mat_features.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        normalized_p = cc.normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        use_aspect = 1 if self.use_aspect_ratio else 0
        use_stats = 1 if self.use_stats else 0
        n_hor_crossings = self.count_crossings if self.use_horizontal_crossings else 0
        n_ver_crossings = self.count_crossings if self.use_vertical_crossings else 0
        hist_size = self.histogram_bins if self.use_2D_histogram else 0

        arg_types = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                     ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.accessmath_feats.get_cc_features.argtypes = arg_types
        self.accessmath_feats.accessmath_feats.get_cc_features.restype = None
        self.accessmath_feats.accessmath_feats.get_cc_features(normalized_p, ConnectedComponent.NormalizedSize, use_aspect,
                                                            use_stats, n_hor_crossings, n_ver_crossings, hist_size,
                                                            self.histogram_scale,mat_features_p)

        return mat_features


    # WARNING: This function is deprecated
    @staticmethod
    def getSURF(cc):
        # SURF extraction
        surf = cv2.SURF(400)
        surfDescriptor = cv2.DescriptorExtractor_create("SURF")

        if cc.normalized == None:
            # create a standar resized (and recenter image)
            cc.normalizeImage(ConnectedComponent.NormalizedSize)

        #...from the normalized...
        key_points  = surf.detect( cc.normalized )
        (key_points, descriptors) = surfDescriptor.compute( cc.normalized, key_points)

        return key_points, descriptors

