#===================================================================
# Attempts to detect keyframes on whiteboard videos by finding
# local minima in frame differences from a small sample of frames
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2015
#
# Modified by:
#   - Kenny Davila (August 23, 2015)
#     - Initial version
#
#===================================================================

import math
import cv, cv2
import numpy as np

from AccessMath.preprocessing.content.binarizer import Binarizer

class StatMotionRemover:
    def __init__(self, segment_stats, background_field="background"):
        self.width = 0
        self.height = 0

        self.frames_differences = None
        self.last_gray_scale = None

        self.segment_stats = segment_stats
        self.bg_field = background_field

        self.last_stat = 0
        self.computed_image = None
        self.last_known = None
        self.last_binary = None

        self.frame_count = 0

        self.output_buffer = None
        self.diff_buffer = None

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frames_differences = []
        self.frame_count = 0

        # grayscale reconstruction
        self.computed_image = None
        self.last_known = None
        # binary reconstruction
        self.last_binary = None
        self.last_stat = 0

        self.last_gray_scale = None

        # compute here the mid_abs and de-compress the background images ...
        for segment in self.segment_stats:
            segment["mid_abs"] = (segment["start_abs"] + segment["end_abs"]) / 2.0

            # decode background (median)
            raw_data = segment[self.bg_field]
            segment["background"] = cv2.imdecode(raw_data, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        self.output_buffer = []
        self.diff_buffer = []

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name


    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        #print("Here: " + str(abs_time))

        self.frame_count += 1

        gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY).astype('int32')

        # check corresponding stat image ...
        # note that video segment processor works with seconds instead of milliseconds
        sec_abs_time = abs_time / 1000.0
        # find current segment ...
        while (self.last_stat + 1 < len(self.segment_stats) and
               self.segment_stats[self.last_stat + 1]["mid_abs"] <= sec_abs_time):
            self.last_stat += 1

        # interpolate background image ...
        # short ...
        if self.last_stat == 0 and sec_abs_time < self.segment_stats[0]["mid_abs"]:
            background = self.segment_stats[0]["background"]

        elif self.last_stat == len(self.segment_stats) - 1:
            background = self.segment_stats[-1]["background"]

        else:
            first_stat = self.segment_stats[self.last_stat]
            next_stat = self.segment_stats[self.last_stat + 1]

            w1 = (sec_abs_time - first_stat["mid_abs"]) / (next_stat["mid_abs"] - first_stat["mid_abs"])
            background = (first_stat["background"] * (1.0 - w1)) + (next_stat["background"] * w1)


        # detect differences using a grid ....
        grid_cols = 32 # 16
        grid_rows = 18  # 9
        col_min_size = int(self.width / grid_cols)
        row_min_size = int(self.height / grid_rows)
        col_rem = self.width % grid_cols
        row_rem = self.height % grid_rows

        threshold_diff = 50
        # 14400 -> 100%
        threshold_strong_region = 0.2 # 10
        threshold_weak_region = 0.001     # original
        # 7200 -> 50% good for strong, harmless strong false positives
        # 3600 -> 25% great for strong, harmless strong false positives (on 01, 15)
        # 2880 -> 20% prettu good! for strong
        # 1800 -> 12.50% still great for strong, still harmless strong false positives (01, 15)
        # 1200 ->  8.33% bad for strong, losses content on lecture 01
        #  720 ->  5.00% bad for strong, losses content on lecture 01

        raw_diff = gray_scale - background

        norm_diff = raw_diff + 128
        norm_diff[norm_diff > 255] = 255
        norm_diff[norm_diff < 0] = 0
        flag, raw_data = cv2.imencode(".png", norm_diff.astype("uint8"))
        self.diff_buffer.append(raw_data)

        raw_diff = np.abs(raw_diff)
        all_diff = (raw_diff > threshold_diff).astype('uint8')
        #print(str(self.frame_count) + "\t" + str(raw_diff.sum()))

        strong_region_mask = np.zeros((grid_rows, grid_cols), dtype=np.bool)
        weak_region_mask = np.zeros((grid_rows, grid_cols), dtype=np.bool)

        #tempo_values = []
        for row in range(grid_rows):
            row_start = row * row_min_size + min(row, row_rem)
            row_end = (row + 1) * row_min_size + min(row + 1, row_rem)

            for col in range(grid_cols):
                col_start = col * col_min_size + min(col, col_rem)
                col_end = (col + 1) * col_min_size + min(col + 1, col_rem)

                region_size = float((col_end - col_start) * (row_end - row_start))

                #mark_region = all_diff[row_start:row_end, col_start:col_end].sum() > threshold_region
                region_sum = all_diff[row_start:row_end, col_start:col_end].sum()

                strong_region_mask[row, col] = region_sum >= region_size * threshold_strong_region
                weak_region_mask[row, col] = region_sum >= region_size * threshold_weak_region

                #if region_sum > threshold_region:
                #    tempo_values.append( (region_sum, round((region_sum / region_size) * 100, 2)) )

        # combine strong and weak mask
        region_mask = np.zeros(gray_scale.shape, dtype=np.bool)
        accepted_strong = np.zeros(gray_scale.shape, dtype=np.bool)
        accepted_weak = np.zeros(gray_scale.shape, dtype=np.bool)
        for row in range(grid_rows):
            row_start = row * row_min_size + min(row, row_rem)
            row_end = (row + 1) * row_min_size + min(row + 1, row_rem)

            for col in range(grid_cols):
                col_start = col * col_min_size + min(col, col_rem)
                col_end = (col + 1) * col_min_size + min(col + 1, col_rem)

                mark = strong_region_mask[row, col]
                if not mark and weak_region_mask[row, col]:
                    # is weak region, mark if has a strong neighbor ...
                    neighborhood = 1
                    for n_row in range(row - neighborhood, row + neighborhood +  1):
                        for n_col in range(col - 1, col + 2):
                            if n_row >= 0 and n_row < grid_rows and n_col >= 0 and n_col < grid_cols:
                                if strong_region_mask[n_row, n_col]:
                                    mark = True

                region_mask[row_start:row_end, col_start:col_end] = mark
                if mark:
                    accepted_strong[row_start:row_end, col_start:col_end] = strong_region_mask[row, col]

                    if not strong_region_mask[row, col]:
                        accepted_weak[row_start:row_end, col_start:col_end] = mark

        inverse_mask = np.logical_not(region_mask)

        #print(self.frame_count)
        #print(tempo_values)

        binary = Binarizer.frameContentBinarization(frame, Binarizer.MethodBackgroundSubstraction)

        #all_diff = region_mask.astype('uint8') * 255 - all_diff * 128
        if self.last_binary is None:
            self.computed_image = background.copy()
            self.last_known = gray_scale.copy()

            self.last_binary = np.zeros(gray_scale.shape, np.uint8)
        else:
            # update regions not covered ...
            self.computed_image[inverse_mask] = gray_scale[inverse_mask]
            self.last_known[inverse_mask] = gray_scale[inverse_mask]

            # do something about the covered ....
            #self.computed_image[region_mask] = 0.0
            self.computed_image[region_mask] = self.last_known[region_mask] * 0.5 + background[region_mask] * 0.5

            self.last_binary[inverse_mask] = binary[inverse_mask]

        #all_diff = self.last_binary

        # add last binary image to output buffer ...
        flag, raw_data = cv2.imencode(".png", self.last_binary)
        self.output_buffer.append((sec_abs_time, raw_data))

        # add to sum image for output
        if self.debug_mode:
            debug_output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # gray scale in blue channel
            debug_output[:,:, 0] = gray_scale
            # selected cells in dark green
            debug_output[accepted_strong, 1] = 192
            debug_output[accepted_weak, 1] = 96
            # white CC
            debug_output[self.last_binary > 0, :] = 255
            # diff in green channel, overwrite blue, allow red to stay...
            debug_output[all_diff > 0, 0] = 0
            #debug_output[all_diff > 0, 1] = 0
            debug_output[all_diff > 0, 2] = 255

            #tempo
            raw_diff = gray_scale - background
            raw_diff[raw_diff > 127] = 127
            raw_diff[raw_diff < -127] = -127
            background =  np.zeros((self.height, self.width, 3), dtype=np.uint8)
            background[raw_diff > 0, 0] = raw_diff[raw_diff > 0] * 2
            background[raw_diff < 0, 2] = np.abs(raw_diff[raw_diff < 0]) * 2


            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(background, debug_output, abs_time)
                #self.debug_frame(background, background, abs_time)

        self.last_gray_scale = gray_scale


    def debug_frame(self, background, processed, abs_time):
        cv2.imwrite(self.debug_out_dir + "/proc_" + self.debug_video_name + "_" + str(self.frame_count) + ".png", processed)
        cv2.imwrite(self.debug_out_dir + "/model_" + self.debug_video_name + "_" + str(self.frame_count) + ".png", background)

    def getWorkName(self):
        return "Motion Remover"

    def finalize(self):
        pass

