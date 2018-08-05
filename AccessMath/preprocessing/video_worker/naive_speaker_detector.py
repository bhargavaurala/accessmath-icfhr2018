import math
import ctypes

import cv2
import numpy as np

from AccessMath.util.misc_helper import MiscHelper
from AccessMath.preprocessing.data.naive_motion_region import NaiveMotionRegion



#===================================================================
# Routines for speaker detection and tracking with the purpose
# of identifying clean frames that could become keyframes of the
# video
#
# By: Kenny Davila 
#     Rochester Institute of Technology
#     2013
#
# Modified by:
#   - Kenny Davila (May 2, 2014)
#     - Added Debug Functionality
#     - Added post processing phase to adjust motion detection
#       based on temporal information
#   - Kenny Davila (May 5, 2014)
#     - Core Functionality move to C for higher performance
#
#===================================================================

#==================================================================
# Detect the approximate location of the speaker at every frame
# on all files. if limit > 0 then only the specified number of frames
# will be processed
#===================================================================

accessmath_lib = ctypes.CDLL('./accessmath_lib.so')

class NaiveSpeakerDetector:
    CELL_JUMP_SIZE = 4
    SPEAKER_MOTION_STD = 3.0
    INNERTIA_WEIGHT = 0.75
    INNERTIA_LENGHT = 3
    MOTION_MARGIN_PERCENTAGE = 0.1

    def __init__(self, change_sensitivity):
        self.change_sensitivity = change_sensitivity
        self.motion_detected = None
        self.debug_mode = False
        self.debug_start = None
        self.debug_end = None
        self.width = None
        self.height = None
        self.debug_temporal = []

    def set_debug_mode(self, active, start_time, end_time):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_temporal = []

    def initialize(self, width, height):
        self.motion_detected = []
        self.width = width
        self.height = height
        self.debug_temporal = []

    def debug_frame(self, frame, last_frame, original_size, region):
        cell_jumps = NaiveSpeakerDetector.CELL_JUMP_SIZE

        #substract...
        diff = np.absolute(np.subtract(frame, last_frame)) > self.change_sensitivity

        #get the positions changed....
        all_y, all_x, c = np.nonzero(diff)

        debug_output = frame.copy()

        #mark points...
        for i in xrange(len(all_y)):
            debug_output[all_y[i], all_x[i], :] = [0, 255, 255]

        #now, mark the box....
        width, height = original_size
        min_x, max_x, min_y, max_y = region.getBoundedLimits(width, height, NaiveSpeakerDetector.SPEAKER_MOTION_STD,
                                                             NaiveSpeakerDetector.SPEAKER_MOTION_STD)
        min_x /= cell_jumps
        max_x /= cell_jumps
        min_y /= cell_jumps
        max_y /= cell_jumps
        cv2.rectangle(debug_output, (min_x, min_y), (max_x, max_y), (0, 0, 255))

        self.debug_temporal.append((region.absolute_time, debug_output.copy()))

        cv2.imwrite("out/speaker_" + str(region.time) + ".png", debug_output)


    def handleFrame_python(self, frame, last_frame, v_index, abs_time, rel_time):
        #do change detection
        #region where changes were detected...
        min_changed_x = self.width + 1
        max_changed_x = -1
        min_changed_y = self.height + 1
        max_changed_y = -1

        cell_jumps = NaiveSpeakerDetector.CELL_JUMP_SIZE
        #....size = width x height
        original_size = (frame.shape[1], frame.shape[0])
        
        #find changes jumping certain number of pixels (subsample pixels)        
        frame = frame[::cell_jumps,::cell_jumps,:].astype('int32')
        last_frame = last_frame[::cell_jumps,::cell_jumps,:].astype('int32')

        #substract...
        diff = np.subtract( frame, last_frame )

        #get the positions changed....
        all_y, all_x, c = np.nonzero( np.absolute(diff) > self.change_sensitivity)


        avg_x = 0.0
        avg_y = 0.0
        changed_count = 0
        cells_changed = []
        
        last_x = -1
        last_y = -1        
        for i in xrange(len(all_y)):
            x = all_x[i] * cell_jumps
            y = all_y[i] * cell_jumps

            if x != last_x or y != last_y:
                #changed ... motion detected...
                if x < min_changed_x:
                    min_changed_x = x
                if x > max_changed_x:
                    max_changed_x = x
                if y < min_changed_y:
                    min_changed_y = y
                if y > max_changed_y:
                    max_changed_y = y

                avg_x += x
                avg_y += y
                cells_changed.append( (x, y) )
                    
                last_x = x
                last_y = y
                changed_count += 1
        
        if changed_count > 0:
            avg_x /= float(changed_count)
            avg_y /= float(changed_count)

            #now get the variance of changed points...
            var_x = 0.0
            var_y = 0.0
            for x, y in cells_changed:
                var_x += (x - avg_x) * (x - avg_x)
                var_y += (y - avg_y) * (y - avg_y)

            var_x /= float(changed_count)
            var_y /= float(changed_count)

            std_x = math.sqrt( var_x )
            std_y = math.sqrt( var_y )
        else:
            std_x = 0.0
            std_y = 0.0

        center_changed = (avg_x, avg_y)
        region_changed = (min_changed_x, max_changed_x, min_changed_y, max_changed_y)
        stds_changed = (std_x, std_y)

        #store results....
        #...creating region...
        region = NaiveMotionRegion(v_index, rel_time, abs_time, changed_count,
                                   center_changed, stds_changed, region_changed)
        #...add to results list...
        self.motion_detected.append(region)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(frame, last_frame, original_size, region)

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        #do change detection

        #....size = width x height
        original_size = (frame.shape[1], frame.shape[0])
        cell_jumps = NaiveSpeakerDetector.CELL_JUMP_SIZE

        frame_p = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        last_frame_p = last_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        boundaries = np.zeros((4), dtype=np.float64)
        boundaries_p = boundaries.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        change_avg = np.zeros((2), dtype=np.float64)
        change_avg_p = change_avg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        change_std = np.zeros((2), dtype=np.float64)
        change_std_p = change_std.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        arg_types = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32,
                     ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                     ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        accessmath_lib.speaker_detection_handle_frame.argtypes = arg_types
        accessmath_lib.speaker_detection_handle_frame.restype = ctypes.c_int32
        count = accessmath_lib.speaker_detection_handle_frame(frame_p, last_frame_p, frame.shape[1], frame.shape[0],
                                                                   frame.shape[2], self.change_sensitivity, cell_jumps,
                                                                   boundaries_p, change_avg_p, change_std_p)

        center_changed = (change_avg[0], change_avg[1])
        region_changed = (boundaries[0], boundaries[1], boundaries[2], boundaries[3])
        stds_changed = (change_std[0], change_std[1])

        #store results....
        #...creating region...
        region = NaiveMotionRegion(v_index, rel_time, abs_time, count,
                              center_changed, stds_changed, region_changed)
        #...add to results list...
        self.motion_detected.append(region)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:

                #find changes jumping certain number of pixels (subsample pixels)
                frame = frame[::cell_jumps,::cell_jumps,:].astype('int32')
                last_frame = last_frame[::cell_jumps,::cell_jumps,:].astype('int32')

                self.debug_frame(frame, last_frame, original_size, region)

    def getWorkName(self):
        return "Speaker Detection"

    def finalize(self):
        pass

    def getMotionDetected(self):
        return self.motion_detected

    #========================================================
    #  Finds the motion with the closest time stamp
    #  after the given time 
    #========================================================
    def findMotionByTime(self, time ):
        #assume that motions array is in cronological order...
        
        init = 0
        end = len(self.motion_detected) - 1
        while init <= end:
            m = int((init + end) / 2)
            if self.motion_detected[m].absolute_time < time:
                init = m + 1
            else:
                end = m - 1

        return init

    #============================================================
    # Find intervals on which the given region is free of
    # obstruction
    #============================================================
    def getNonblockedIntervals(self, region_box, max_width, max_height, init_index, end_time):
        #Find a frame where the found region has no motion around

        pos = init_index
        blocked_list = []

        while pos < len(self.motion_detected) and \
              self.motion_detected[pos].absolute_time < end_time:

            #check motion[pos] to see if main_region is blocked
            blocked = self.motion_detected[pos].isBlockingRegion(region_box, max_width, max_height, 3.0, 3.0)

            #add to boolean list
            blocked_list.append(blocked)

            pos += 1

        #now find the intervals where it is not obstruded...
        intervals = MiscHelper.findBooleanIntervals( blocked_list, False )

        return intervals

    #==============================================================
    #   Select frames where speaker is believed to be absent
    #   from the image based on the motion detection analysis
    #==============================================================
    def selectMotionlessFrames(self):
        #maximum motion allowed
        threshold = 0

        booleans = []
        for idx, m in enumerate(self.motion_detected):
            booleans.append( m.count_changes <= threshold )

        intervals = MiscHelper.findBooleanIntervals( booleans, True)

        #only consider intervals of at least 3 frames
        candidates = []
        for ini, end in intervals:
            #check....
            if end - ini >= 2:
                #pick the frame in the middle of the interval
                middle = self.motion_detected[int((end + ini) / 2.0)]
                candidates.append( (middle.video_index, middle.time) )

        return candidates

    def postProcess(self):
        #adjust the average and std of motion of points for
        #smoother speaker detection

        previous_on_board = False
        previous_mins_x = []
        previous_maxs_x = []

        for idx, motion in enumerate(self.motion_detected):
            curr_box = motion.getBoundedLimits(self.width, self.height, NaiveSpeakerDetector.SPEAKER_MOTION_STD,
                                               NaiveSpeakerDetector.SPEAKER_MOTION_STD)
            curr_min_x, curr_max_x, curr_min_y, curr_max_y = curr_box


            #adjust vertical...
            min_margin = int(self.width * NaiveSpeakerDetector.MOTION_MARGIN_PERCENTAGE)
            max_margin = int(self.width * (1.0 - NaiveSpeakerDetector.MOTION_MARGIN_PERCENTAGE))
            if min_margin <= motion.center_x <= max_margin:
                #speaker is in front of the board...

                #If top is found to require adjustments, it should be adjusted here...

                #The bottom of rectangle should be at bottom of frame...
                if curr_max_y < self.height:
                    #adjust...
                    n_std_y = (self.height - curr_min_y) / (2.0 * NaiveSpeakerDetector.SPEAKER_MOTION_STD)
                    n_center_y = curr_min_y + NaiveSpeakerDetector.SPEAKER_MOTION_STD * n_std_y

                    motion.center_y = n_center_y
                    motion.std_y = n_std_y
                    motion.max_y = self.height

                #adjust horizontal limits...
                #consider inertia and current limits...
                if len(previous_mins_x) > 0:
                    #get the inertia...
                    n_values = float(len(previous_mins_x))
                    inertia_min_x = (sum(previous_mins_x) / n_values)
                    inertia_max_x = (sum(previous_maxs_x) / n_values)

                    #new boundaries based on inertia
                    new_min_x = ((inertia_min_x * NaiveSpeakerDetector.INNERTIA_WEIGHT) +
                                 (curr_min_x * (1.0 - NaiveSpeakerDetector.INNERTIA_WEIGHT)))
                    new_max_x = ((inertia_max_x * NaiveSpeakerDetector.INNERTIA_WEIGHT) +
                                 (curr_max_x * (1.0 - NaiveSpeakerDetector.INNERTIA_WEIGHT)))

                    #adjust ... new boundaries must contain current boundaries..
                    new_min_x = min(new_min_x, curr_min_x)
                    new_max_x = max(new_max_x, curr_max_x)

                    #compute new boundaries....
                    n_std_x = (new_max_x - new_min_x) / (2.0 * NaiveSpeakerDetector.SPEAKER_MOTION_STD)
                    n_center_x = new_min_x + NaiveSpeakerDetector.SPEAKER_MOTION_STD * n_std_x

                    #set new boundaries...
                    motion.center_x = n_center_x
                    motion.std_x = n_std_x
                    motion.min_x = min(motion.min_x, new_min_x)
                    motion.max_x = min(motion.max_x, new_max_x)


                previous_on_board = True
                previous_mins_x.append(curr_min_x)
                previous_maxs_x.append(curr_max_x)

                #keep "tail" of moves only as long as required...
                if len(previous_mins_x) > NaiveSpeakerDetector.INNERTIA_LENGHT:
                    del previous_mins_x[0]
                    del previous_maxs_x[0]
            else:
                #restart...
                previous_mins_x = []
                previous_maxs_x = []
                previous_on_board = False

            #for debugging....
            if self.debug_mode and len(self.debug_temporal) > 0:
                #check if next frame for debug...
                next_abs_time, next_frame =  self.debug_temporal[0]
                if motion.absolute_time == next_abs_time:
                    #draw the box...
                    curr_box = motion.getBoundedLimits(self.width, self.height, NaiveSpeakerDetector.SPEAKER_MOTION_STD,
                                                       NaiveSpeakerDetector.SPEAKER_MOTION_STD)
                    curr_min_x, curr_max_x, curr_min_y, curr_max_y = curr_box

                    curr_min_x /= NaiveSpeakerDetector.CELL_JUMP_SIZE
                    curr_max_x /= NaiveSpeakerDetector.CELL_JUMP_SIZE
                    curr_min_y /= NaiveSpeakerDetector.CELL_JUMP_SIZE
                    curr_max_y /= NaiveSpeakerDetector.CELL_JUMP_SIZE
                    cv2.rectangle(next_frame, (int(curr_min_x), int(curr_min_y)), (int(curr_max_x), int(curr_max_y)),
                                  (0, 255, 0))

                    cv2.imwrite("out/speaker_post_" + str(next_abs_time) + ".png", next_frame)

                    #remove....
                    del self.debug_temporal[0]


