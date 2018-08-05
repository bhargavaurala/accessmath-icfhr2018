import cv, cv2
import numpy as np
from scipy import stats
import cPickle

#===================================================================
# Routines for simple segment analysis based on basic statistics
# obtained from the block
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2015
#
# Modified by:
#   - Kenny Davila (Aug, 2015)
#
#===================================================================

class SimpleSegmentAnalyzer:
    def __init__(self):
        self.width = None
        self.height = None
        self.segments = None

        self.debug_mode = False
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height
        self.segments = []

    def set_debug_mode(self, mode, out_dir, video_name):
        self.debug_mode = mode
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    def handleFrame(self, frame, last_frame, n_block, v_index, abs_time, rel_Time):
        #print(str(abs_time) + "\t" + str(v_index) + "\t" + str(rel_Time))
        raise Exception("Class SimpleSegmentAnalyzer not designed for sequential processing")

    def handleBlock(self, frames, n_block, start_abs, end_abs):
        if len(frames) == 0:
            return

        # print(str(n_block) + "\t" + str(start_abs) + " - " + str(end_abs) + "\t" + str(len(frames)))

        # first, obtain the gray scale images ...
        height = frames[0].shape[0]
        width = frames[0].shape[1]

        all_images = np.zeros((len(frames), height, width), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            all_images[idx, :,:] = cv2.cvtColor(frame, cv.CV_RGB2GRAY)

        """
        out_file = open("sample_segment_" + self.debug_video_name + "_" + str(n_block) + ".dat", "wb")
        cPickle.dump(all_images, out_file, cPickle.HIGHEST_PROTOCOL)
        out_file.close()
        """

        if self.debug_mode:
            print("Computing Median")

        median_image = np.median(all_images, 0)
        """
        if self.debug_mode:
            print("Computing Mode")
        mode_image, mode_counts = stats.mode(all_images, 0)
        mode_image = mode_image[0, :, :]

        if self.debug_mode:
            print("Computing Min")
        min_image = np.min(all_images, 0)
        if self.debug_mode:
            print("Computing Max")
        max_image = np.max(all_images, 0)
        if self.debug_mode:
            print("Computing Mean")
        avg_image = np.mean(all_images, 0)
        """

        """
        segment_info = {
            "median": median_image,
            "mode": mode_image,
            "min": min_image,
            "max": max_image,
            "avg": avg_image,
        }
        """
        segment_info = {
            "n_block": n_block,
            "start_abs": start_abs,
            "end_abs": end_abs,
            "median": median_image,
        }

        self.segments.append(segment_info)

        # debug ...
        if self.debug_mode:
            common_prefix = self.debug_out_dir + "/" + self.debug_video_name + "_"
            common_sufix = "_" + str(n_block) + ".png"

            cv2.imwrite(common_prefix + "median" + common_sufix, median_image)
            """
            cv2.imwrite(common_prefix + "mode" + common_sufix, mode_image)
            cv2.imwrite(common_prefix + "min" + common_sufix, min_image)
            cv2.imwrite(common_prefix + "max" + common_sufix, max_image)
            cv2.imwrite(common_prefix + "avg" + common_sufix, avg_image)
            """

    def getWorkName(self):
        return "Simple Segment Analyzer"


    def finalize(self):
        pass

    def postProcess(self):
        #Any additional post processing
        pass
