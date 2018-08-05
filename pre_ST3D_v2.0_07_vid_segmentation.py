
#============================================================================
# Preprocessing Model for ST3D indexing - V 1.1
#
# Kenny Davila
# - Created:  March 19, 2017
#
#============================================================================

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.content.helper import Helper
from AccessMath.preprocessing.content.video_segmenter import VideoSegmenter
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

def process_input(process, input_data):

    # configuration overrides ...
    if "conf_w" in process.params:
        Parameters.VSeg_Conf_weights = int(process.params["conf_w"])

    if "conf_t" in process.params:
        Parameters.VSeg_Conf_weights_time = int(process.params["conf_t"]) > 0

    if Parameters.VSeg_method == 2:
        frame_times, frame_indices, compressed_frames = input_data[0]
        group_ages, conflicts = input_data[1]
    else:
        frame_times, frame_indices, compressed_frames = input_data

    debug_mode = True

    # 1) decompress all images ...
    print("Decompressing input...")
    all_binary = Helper.decompress_binary_images(compressed_frames)

    # 2) Computing all sums...
    print("Computing sums...")
    all_sums = VideoSegmenter.compute_binary_sums(all_binary)

    # 3) Getting the intervals ...
    if Parameters.VSeg_method == 2:
        # using conflicts
        # now, compute ideal intervals based on conflicts
        save_file_prefix = process.img_dir + "/group_segment_" + process.current_lecture.title + "_"
        intervals = VideoSegmenter.video_segments_from_group_conflicts(len(all_binary), group_ages, conflicts,
                                                                       Parameters.VSeg_Conf_min_conflicts,
                                                                       Parameters.VSeg_Conf_min_split,
                                                                       Parameters.VSeg_Conf_min_len,
                                                                       Parameters.VSeg_Conf_weights,
                                                                       Parameters.VSeg_Conf_weights_time,
                                                                       save_file_prefix)
    else:
        # using sums
        # minimum size of a leaf in the Regression Decision Tree
        leaf_min = int(math.ceil(Parameters.VSeg_Sum_min_segment * Parameters.Sampling_FPS))

        intervals = VideoSegmenter.video_segments_from_sums(all_sums, leaf_min,Parameters.VSeg_Sum_min_erase_ratio)
        print("Erasing Events: ")
        print(intervals)

    # 4) Debug output...
    if debug_mode:

        y = np.array(all_sums)

        # Plot the results
        fig = plt.figure(figsize=(8,6),dpi=200)

        ax1 = fig.add_subplot(111)

        max_y_value = y.max() * 1.10

        X = np.arange(len(all_sums))
        ax1.fill_between(X, y, facecolor="#7777DD", alpha=0.5)

        if Parameters.VSeg_method == 2:
            plt.title("Conflict Minimization Video Segmentation")
        else:
            eval_X = np.arange(len(all_sums)).reshape(len(all_sums), 1)

            regressor = VideoSegmenter.create_regresor_from_sums(all_sums, leaf_min)
            y_1 = regressor.predict(eval_X)
            color = "#2222FF"
            plt.plot(eval_X, y_1, c=color, linewidth=2)
            #ax1.fill_between(X, y_1, c=color, linewidth=2, alpha=0.5)
            plt.title("Decision Tree Regression Video Segmentation")


        for start_idx, end_idx  in intervals:
            first_x =  X[start_idx]
            last_x = X[end_idx]

            plt.plot(np.array([first_x, first_x]), np.array([0, max_y_value]), c="g", linewidth=1)
            plt.plot(np.array([last_x, last_x]), np.array([0, max_y_value]), c="r", linewidth=1)

        plt.xlabel("data")
        plt.ylabel("target")

        #plt.legend()

        out_filename = process.img_dir + "/intervals_" + str(Parameters.VSeg_method) + "_" + process.current_lecture.title + ".png"
        plt.savefig(out_filename, dpi=200)

        plt.close()

    print("Total intervals: " + str(len(intervals)))

    return intervals

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    inputs = [Parameters.Output_CCReconstructed, Parameters.Output_CCConflicts]

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], inputs, Parameters.Output_Vid_Segment)
    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")

if __name__ == "__main__":
    main()

