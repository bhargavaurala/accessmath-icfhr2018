#===================================================================
# Binarization and extraction of raw connected components (CC) from
# a given video
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2015
#
# Modified by:
#   - Kenny Davila (June 23, 2015)
#     - Initial version
#
#===================================================================


import cv, cv2
import numpy as np
from AccessMath.preprocessing.content.board_detector import BoardDetector
from AccessMath.preprocessing.content.binarizer import Binarizer
from AccessMath.preprocessing.content.labeler import Labeler

class RawCCExtractor:
    def __init__(self, binarization_method_id):
        self.bin_method = binarization_method_id

        self.width = 0
        self.height = 0

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""
        self.debug_frame_count = 0

        self.all_components = []

        self.max_previous = 100
        self.next_offset = 0
        self.previous = None

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_title):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_title

    def initialize(self, width, height):
        self.width = width
        self.height = height
        self.all_binary = []
        self.debug_frame_count = 0

        self.next_offset = 0
        self.previous = np.zeros((self.max_previous, height, width), np.uint8)

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        #print("Here: " + str(abs_time))

        self.debug_frame_count += 1
        """
        binary = Binarizer.regionBinarization(frame, self.bin_method)
        components = Labeler.extractConnectedComponents(binary)

        self.all_components.append(components)
        """
        """
        if not self.debug_frame_count in [3, 4, 12, 29, 30, 60, 220, 246]:
            return
        """

        gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
        self.previous[self.next_offset,:,:] = gray_scale
        self.next_offset += 1
        if self.next_offset >=  self.max_previous:
            self.next_offset = 0

        median_image = np.median(self.previous[0:min(self.debug_frame_count, self.max_previous), :, :], 0)

        mask = np.absolute(gray_scale - median_image) > 25
        binary = np.zeros(frame.shape, frame.dtype)
        binary[mask,0] = frame[mask, 0]
        binary[mask,1] = frame[mask, 1]
        binary[mask,2] = frame[mask, 2]

        #binary = BoardDetector.detect(frame, BoardDetector.MethodDefault)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(frame, binary, abs_time)


    def debug_frame(self, frame, binary, abs_time):
        #cv2.imwrite(self.debug_out_dir + "/bin_" + str(abs_time) + ".png",  frame)
        cv2.imwrite(self.debug_out_dir + "/bin_" + self.debug_video_name + "_" + str(self.debug_frame_count) + ".png", binary)
        cv2.imwrite(self.debug_out_dir + "/frame_" + self.debug_video_name + "_" + str(self.debug_frame_count) + ".png", frame)

    def getWorkName(self):
        return "RAW CC EXTRACTION"

    def finalize(self):
        pass
