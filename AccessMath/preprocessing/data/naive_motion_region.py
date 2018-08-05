
#===============================================================
# CLASS USED TO STORE INFORMATION ABOUT MOTION DETECTED ON A
# FRAME OF THE VIDEO THAT CONTAINS THE SPEAKER
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#===============================================================

class NaiveMotionRegion:
    #constructor.....
    def __init__(self, video_idx, time, absolute_time, count, center, stds, limits):
        self.video_index = video_idx
        self.time = time
        self.absolute_time = absolute_time
        self.count_changes = count

        self.center_x = center[0]
        self.center_y = center[1]

        self.std_x = stds[0]
        self.std_y = stds[1]

        self.min_x = limits[0]
        self.max_x = limits[1]
        self.min_y = limits[2]
        self.max_y = limits[3]

    #=================================================
    # Gets the limits of the region bounded by a
    # maximum distance from the average given by
    # number of standard deviations
    #=================================================
    def getBoundedLimits(self, width, height, max_std_x = 2.0, max_std_y = 2.0):
        #....x....
        b_min_x = max(self.min_x, int(max(0, self.center_x - self.std_x * max_std_x)))
        b_max_x = min(self.max_x, int(min(width - 1, self.center_x + self.std_x * max_std_x)))
        #....y....
        b_min_y = max(self.min_y, int(max(0, self.center_y - self.std_y * max_std_y)))
        b_max_y = min(self.max_y, int(min(height - 1, self.center_y + self.std_y * max_std_y)))

        return (b_min_x, b_max_x, b_min_y, b_max_y)

    def isBlockingRegion(self, region_box, max_width, max_height, max_std_x=2.0, max_std_y=2.0):
        if self.count_changes == 0:
            # assume free of blockage...
            return False
        else:
            # get bounded limits...
            limits = self.getBoundedLimits( max_width, max_height, max_std_x, max_std_y)

            # check if boxes overlap
            return (limits[0] < region_box[1] and
                    region_box[0] < limits[1] and
                    limits[2] < region_box[3] and
                    region_box[2] < limits[3])

    def __str__(self):
        return "Region <time: " + str(self.time) + "(" + str(self.video_index) + ")" + \
               ", (" + str(self.center_x) + ", " + str(self.center_y) + ") - " + str(self.count_changes)
