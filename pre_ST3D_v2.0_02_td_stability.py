
import sys

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.config.parameters import Parameters

from AccessMath.preprocessing.content.bbox_stability_estimator import BBoxStabilityEstimator

def process_input(process, input_data):
    detection_results = input_data[0]

    # minimum area ratio between union of two boxes areas and the area of the minimum bounding box containing both
    min_combined_box_ratio = Parameters.TDStab_min_comb_box_ratio
    min_temporal_box_IOU = Parameters.TDStab_min_temp_box_IOU        # minimum spatial IOU to combine boxes across time
    min_temporal_gap = Parameters.TDStab_max_temporal_gap
    min_TD_confidence = Parameters.TDStab_min_confidence # 0.65

    # TODO: this should come from detection_results, they should include source resolution!
    width = 1920
    height = 1080

    bbox_stability = BBoxStabilityEstimator(width, height, min_combined_box_ratio, min_temporal_box_IOU,
                                            min_temporal_gap, True)
    
    # sorted key-frames
    keyframe_ids = sorted(list(detection_results.keys()))

    # extract and group detected bounding boxes ...
    frame_times = []
    total_eliminated = 0
    for keyframe_id in keyframe_ids:
        frame_results = detection_results[keyframe_id]

        frame_times.append(frame_results["abs_time"])

        accepted_bboxes = []
        for bbox_idx, bbox in enumerate(frame_results["bboxes"]):
            if frame_results["confidences"][bbox_idx] >= min_TD_confidence:
                accepted_bboxes.append(bbox)
            else:
                total_eliminated += 1

        bbox_stability.add_frame(accepted_bboxes)

    msg = "\nA total of {0:d} boxes with confidence lower than {1:f} were discarded"
    print(msg.format(total_eliminated, min_TD_confidence))

    return frame_times, keyframe_ids, bbox_stability

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    input_prefix = Parameters.Output_TextDetection
    output_prefix = Parameters.Output_TDStability
    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], input_prefix, output_prefix)

    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")

if __name__ == "__main__":
    main()
