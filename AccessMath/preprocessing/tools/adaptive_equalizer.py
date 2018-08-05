
#================================================================
# FUNCTIONS FOR ADAPTIVE HISTOGRAM EQUALIZATION
#
# Original code by: Benjamin Sommer
# URL: http://weblog.benjaminsommer.com/blog/2012/05/25/histogram-equalization/
#
# Changes made:
#  - Original code used Brute Force, new one is Tile based
#     - This version uses interpolation of patches
#     - Computationally more efficient
#     - Results are very similar to Matlab's results adapthisteq
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013-2015
#================================================================

import math
import numpy as np
import ctypes

class AdaptiveEqualizer:
    accessmath_lib = ctypes.CDLL('./accessmath_lib.so')

    #======================================================
    # Compute the histogram of intensities for a given
    # image, grayscale values must be in range [0,255]
    #======================================================
    @staticmethod
    def histogram(grayscale, bins=256):
        h = np.zeros((bins))
        for i in xrange(grayscale.shape[0]):
            for j in xrange(grayscale.shape[1]):
                h[grayscale[i][j]] += 1
        return h

    #========================================================
    # Compute the local histogram of intensities for a given
    # image around a given pivot and with the maximum
    # distance specified.
    #=========================================================
    @staticmethod
    def localHistogram(grayscale, pivot, radius, bins=256):
        h = np.zeros((bins))
        for i in xrange(max(pivot[0]-radius,0), min(pivot[0]+radius,grayscale.shape[0])):
            for j in xrange(max(pivot[1]-radius,0), min(pivot[1]+radius,grayscale.shape[1])):
                h[grayscale[i][j]] += 1
        return h

    #========================================================
    # Compute the local histogram of intensities for a given
    # image over the specified image patch.
    #=========================================================
    @staticmethod
    def regionHistogram(grayscale, min_x, max_x, min_y, max_y, bins=256 ):
        h = np.zeros((bins))
        for x in xrange(max(min_x, 0), min(max_x + 1, grayscale.shape[1])):
            for y in xrange(max(min_y, 0), min(max_y + 1, grayscale.shape[0])):
                h[grayscale[y][x]] += 1

        return h

    #========================================================
    # Compute the cumulative distribution from a given
    # histogram of intensities
    #=========================================================
    @staticmethod
    def cumulativeDistr(histogram, slope_max = 0):
        d = np.zeros((histogram.size))
        c = 0
        for i in xrange(histogram.size):
            c += histogram[i]
            d[i] = c

        d /= c

        if (slope_max > 0):
            dh = 0
            for i in xrange(d.size-1):
                dh += max(d[i+1]-d[i]-dh-slope_max,0)
                d[i+1] -= dh

        return d

    #========================================================
    # Compute the cumulative distribution from a given
    # histogram of intensities
    #=========================================================
    @staticmethod
    def equalizeHistogram(grayscale):
        h = AdaptiveEqualizer.histogram(grayscale)
        d = AdaptiveEqualizer.cumulativeDistr(h)
        img = np.zeros(grayscale.shape)
        for i in xrange(grayscale.shape[0]):
            for j in xrange(grayscale.shape[1]):
                img[i][j] = d[grayscale[i][j]]
        return img

    #========================================================
    # Adaptive Histogram Equalization (Python Implementation)
    #=========================================================
    @staticmethod
    def adapthisteq_python(grayscale, slope, grid_x, grid_y):
        img = np.zeros(grayscale.shape)

        #generate tile histograms....
        min_size_y = int(math.floor( grayscale.shape[0] / grid_y ))
        min_size_x = int(math.floor( grayscale.shape[1] / grid_x ))


        #create structures...
        x_limits = [ None for x in range(grid_x) ]
        y_limits = [ None for y in range(grid_y) ]
        histograms = [ [ None for x in range(grid_x) ] for y in range(grid_y) ]
        distributions = [ [ None for x in range(grid_x) ] for y in range(grid_y) ]

        #calculate boundaries ...
        # ... x ...
        last_x = 0
        for rx in range(grid_x):
            w = min_size_x + (1 if rx < grayscale.shape[1] % grid_x else 0)
            end_x = last_x + w - 1

            x_limits[rx] = (last_x, end_x, int(round((last_x + end_x) / 2.0)))

            last_x = end_x + 1

        #.... y ...
        last_y = 0
        for ry in range(grid_y):
            h = min_size_y + (1 if ry < grayscale.shape[0] % grid_y else 0)
            end_y = last_y + h - 1

            y_limits[ry] = (last_y, end_y, int(round((last_y + end_y) / 2.0)))

            last_y = end_y + 1

        #calculate histograms and CDF's
        for rx in range(grid_x):
            last_x, end_x, m_x = x_limits[rx]

            for ry in range(grid_y):
                last_y, end_y, m_y = y_limits[ry]

                #calculate the histograms of the grid...
                hist = AdaptiveEqualizer.regionHistogram( grayscale, last_x, end_x, last_y, end_y, bins=256 )
                histograms[ry][rx] = hist

                #calculate the CDF
                dist = AdaptiveEqualizer.cumulativeDistr(hist, slope)

                #check min value and max value to center distribution
                offset = (1.0 - (dist[255] - dist[0])) / 2.0
                #now center...
                for val in range(len(dist)):
                    dist[val] += offset

                distributions[ry][rx] = dist

        #now use interpolation
        current_x = 0
        for x in range(grayscale.shape[1]):
            #update current cell x
            if x > x_limits[current_x][1]:
                current_x += 1

            current_y = 0
            for y in range(grayscale.shape[0] ):
                #update current cell y
                if y > y_limits[current_y][1]:
                    current_y += 1

                tone = grayscale[y][x]

                #check current case ...
                if current_x == 0 and x < x_limits[0][2]:
                    #on first tile, before middle pixel

                    if current_y == 0 and y < y_limits[0][2]:
                        #exactly the top-left corner... use just one patch
                        img[y][x] = distributions[0][0][grayscale[y][x]]
                    elif current_y == grid_y - 1 and y >= y_limits[grid_y - 1][2]:
                        #exactly the bottom-left corner... use just one patch
                        img[y][x] = distributions[grid_y - 1][0][tone]
                    else:
                        #somewhere between the top and bottom...
                        y0 = current_y - (1 if y <= y_limits[current_y][2] else 0)
                        y1 = y0 + 1

                        wy1 = (y - y_limits[y0][2]) / float(y_limits[y1][2] - y_limits[y0][2])

                        d0 = distributions[y0][0]
                        d1 = distributions[y1][0]

                        #linear interpolation
                        img[y][x] = d0[tone] * (1.0 - wy1) + d1[tone] * (wy1)
                elif current_x == grid_x - 1 and x >= x_limits[grid_x - 1][2]:
                    #on the last tile, after middle pixel
                    if current_y == 0 and y < y_limits[0][2]:
                        #exactly the top-right corner... use just one patch
                        img[y][x] = distributions[0][grid_x - 1][grayscale[y][x]]
                    elif current_y == grid_y - 1 and y >= y_limits[grid_y - 1][2]:
                        #exactly the bottom-right corner... use just one patch
                        img[y][x] = distributions[grid_y - 1][grid_x - 1][tone]
                    else:
                        #somewhere between the top and bottom...
                        y0 = current_y - (1 if y <= y_limits[current_y][2] else 0)
                        y1 = y0 + 1

                        wy1 = (y - y_limits[y0][2]) / float(y_limits[y1][2] - y_limits[y0][2])

                        d0 = distributions[y0][grid_x - 1]
                        d1 = distributions[y1][grid_x - 1]

                        #linear interpolation
                        img[y][x] = d0[tone] * (1.0 - wy1) + d1[tone] * (wy1)
                elif current_y == 0 and y < y_limits[0][2]:
                    #vertically on the top tile...
                    #horizontally ... somewhere in the middle...
                    x0 = current_x - (1 if x <= x_limits[current_x][2] else 0)
                    x1 = x0 + 1

                    wx1 = (x - x_limits[x0][2]) / float(x_limits[x1][2] - x_limits[x0][2])

                    d0 = distributions[0][x0]
                    d1 = distributions[0][x1]

                    #linear interpolation
                    img[y][x] = d0[tone] * (1.0 - wx1) + d1[tone] * (wx1)
                elif current_y == grid_y - 1 and y >= y_limits[grid_y - 1][2]:
                    #vertically on the bottom tile...
                    #horizontally ... somewhere in the middle...
                    x0 = current_x - (1 if x <= x_limits[current_x][2] else 0)
                    x1 = x0 + 1

                    wx1 = (x - x_limits[x0][2]) / float(x_limits[x1][2] - x_limits[x0][2])

                    d0 = distributions[grid_y - 1][x0]
                    d1 = distributions[grid_y - 1][x1]

                    #linear interpolation
                    img[y][x] = d0[tone] * (1.0 - wx1) + d1[tone] * (wx1)
                else:
                    #in the middle... complete bilinear interpolation...
                    x0 = current_x - (1 if x <= x_limits[current_x][2] else 0)
                    x1 = x0 + 1
                    wx1 = (x - x_limits[x0][2]) / float(x_limits[x1][2] - x_limits[x0][2])

                    y0 = current_y - (1 if y <= y_limits[current_y][2] else 0)
                    y1 = y0 + 1
                    wy1 = (y - y_limits[y0][2]) / float(y_limits[y1][2] - y_limits[y0][2])

                    d00 = distributions[y0][x0]
                    d01 = distributions[y0][x1]
                    d10 = distributions[y1][x0]
                    d11 = distributions[y1][x1]

                    img[y][x] = d00[tone] * (1.0 - wy1) * (1.0 - wx1) + \
                                d01[tone] * (1.0 - wy1) * (wx1) + \
                                d10[tone] * wy1 * (1.0 - wx1) + \
                                d11[tone] * wy1 * wx1

                #Un-comment to see the tiles....
                #img[y][x] = distributions[current_y][current_x][grayscale[y][x]]

        return img

    #========================================================
    # Adaptive Histogram Equalization (call C Implementation)
    #=========================================================
    @staticmethod
    def adapthisteq(grayscale, slope, grid_x, grid_y):
        height = grayscale.shape[0]
        width = grayscale.shape[1]

        grayscale_p = grayscale.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        result = np.zeros(grayscale.shape, dtype=np.uint8)
        result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # call C implementation ...
        arg_types = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32, ctypes.c_double,
                     ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8)]

        AdaptiveEqualizer.accessmath_lib.adapthisteq.argtypes = arg_types
        AdaptiveEqualizer.accessmath_lib.adapthisteq.restype = ctypes.c_int32
        AdaptiveEqualizer.accessmath_lib.adapthisteq(grayscale_p, width, height, slope, grid_x, grid_y, result_p)

        return result
