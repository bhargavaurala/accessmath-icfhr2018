
#============================================================================
# Preprocessing Model for ST3D indexing - V 1.1
#
# Kenny Davila
# - Created:  March 18, 2017
#
#============================================================================

import sys
import numpy as np

from AccessMath.preprocessing.content.bbox_stability_estimator import BBoxStabilityEstimator
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.config.parameters import Parameters

def process_input(process, input_data):
    frame_times, frame_indices, estimator = input_data

    # TODO: get these from parameters
    max_gap = Parameters.TDStab_max_temporal_gap # (85 for CC)
    min_times = Parameters.TDStab_min_times
    t_window = Parameters.TDGroup_temporal_window

    refinement_type = BBoxStabilityEstimator.RefinePerBoxPerSegment

    # bin_threshold = 0.5
    # group_use_global_img = False

    # 1) De-compress binarized frames ....
    # binary_frames = Helper.decompress_binary_images(compressed_binary)
    binary_frames = None
    compressed_binary = None

    # remove unstable elements
    # 2) Split CC with large gaps
    # (the threshold is considered in the previous step, but this is applied just in case that
    #  a lower threshold is selected after the initial processing was completed)
    print("Splitting Bounding Boxes with large gap ... ")
    count = estimator.split_stable_bboxes_by_gaps(max_gap, min_times)
    print("Total Bounding Boxes split: " + str(count))

    # 3) Get the list of stable CC
    print("Computing stable Bounding boxes")
    stable_idxs = estimator.get_stable_bbox_idxs(min_times)

    n_objects = len(estimator.unique_bbox_objects)
    n_stable = len(stable_idxs)
    n_raw_objects = estimator.get_raw_bbox_count()

    print("Raw Bounding-Box count: " + str(n_raw_objects))
    print("Unique Bounding-Box Count: " + str(n_objects))
    print("Stable Bounding-Box Count: " + str(n_stable))

    # 4) identify overlapping Bounding Boxes
    print("Computing Stable overlapping")

    # t_window = Parameters.CCGroup_temporal_window
    overlapping_stable_info = estimator.compute_overlapping_stable_bboxes(stable_idxs, t_window)

    time_overlapping_bboxes, total_intersections, all_overlapping_bboxes = overlapping_stable_info

    inter_counts = np.array([len(x) for x in time_overlapping_bboxes])
    hist, bin_edges = np.histogram(inter_counts, 10)

    print("")
    print("Total intersections found: " + str(total_intersections))
    print("Intersection histogram:")
    print(bin_edges)
    print(hist)

    # 5) Use overlapping information to compute groups of Bounding boxes
    bbox_groups, group_idx_per_bbox = estimator.compute_groups(stable_idxs, time_overlapping_bboxes)
    n_groups = len(bbox_groups)
    print("Final count of groups: " + str(n_groups))
    print("Final count of non-empty groups: " + str(sum([1 for x in bbox_groups if len(x) > 0])))

    # 6) Get temporal information for groups
    print("Computing ages for groups")
    group_ages, groups_per_frame = estimator.compute_groups_temporal_information(bbox_groups)

    # 7) Find bboxes for stable bboxes and their groups ...
    # (NOTE THAT THIS FUNCTION RETURNS XYXY boundaries unlike
    print("Computing refined bboxes for groups")
    refined_bboxes, groups_bboxes, refined_per_group = estimator.refine_bboxes(bbox_groups, group_ages, refinement_type)

    # 8) Use refined bboxes to generate refined detection results per frame ...
    print("Generating refined bboxes per frame")
    #save_file_prefix = process.img_dir + "/refined_bboxes_" + process.current_lecture.title + "_"
    save_file_prefix = None
    refined_per_frame = estimator.refined_per_frame(bbox_groups, groups_per_frame, group_ages, refined_per_group,
                                                    save_file_prefix, min_times)

    # 9) Use the refined boxes to reconstruct the text detection results ...
    refined_detection_results = {}
    for offset, frame_idx in enumerate(frame_indices):
        refined_detection_results[frame_idx] = {
            "bboxes": refined_per_frame[offset],
            "abs_time": frame_times[offset],
        }

    return (refined_detection_results, )

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    input_prefix = Parameters.Output_TDStability
    output_prefixes = Parameters.Output_TDRefined
    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], input_prefix,
                               output_prefixes)
    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")

if __name__ == "__main__":
    main()
