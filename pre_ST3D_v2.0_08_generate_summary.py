#============================================================================
# Preprocessing Model for ST3D indexing - V 1.1
#
# Kenny Davila
# - Created:  March 18, 2017
#
#============================================================================

import sys
import math

from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.content.helper import Helper
from AccessMath.preprocessing.content.keyframe_extractor import KeyframeExtractor
from AccessMath.preprocessing.content.keyframe_exporter import KeyframeExporter
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

def process_input(process, input_data):
    st3D = input_data[0]
    video_segments = input_data[1]

    # parameters

    # 1) extract image per video segment
    keyframes, cc_times = KeyframeExtractor.GenerateFromST3DForIntervals(st3D, video_segments)

    # 2) Save summary ....
    database = process.database
    lecture = process.current_lecture

    # TODO: FIX THIS!
    output_prefix = database.output_summaries + "/text_detection_" + database.name + "_" + lecture.title.lower()
    print("Saving data to: " + output_prefix)

    idx_intervals = []
    time_intervals = []
    summary_times = []
    summary_indices = []

    # convert the logical frame indices of intervals to their actual absolute frame indices in the video.
    # also expand boundaries to remove gap between video segments
    last_start = 0
    last_time_start = 0
    for idx, (segment_start, segment_end) in enumerate(video_segments):
        # absolute times
        frame_end = st3D.frame_indices[segment_end]
        time_end = st3D.frame_times[segment_end]

        if idx + 1 < len(video_segments):
            # use middle point between original interval end and next original interval start (remove the gap)
            next_frame_start = st3D.frame_indices[video_segments[idx + 1][0]]
            next_time_start = st3D.frame_times[video_segments[idx + 1][0]]
            interval_end = int((frame_end + next_frame_start) / 2)
            time_interval_end = (time_end + next_time_start) / 2.0
        else:
            # last element, no gap to reduce ...
            interval_end = frame_end
            time_interval_end = time_end

        idx_intervals.append((last_start, interval_end))
        time_intervals.append((last_time_start, time_interval_end))
        last_start = interval_end
        last_time_start = time_interval_end

        # still use the last frame of the original interval  for index and time  ...
        summary_indices.append(frame_end)
        summary_times.append(st3D.frame_times[segment_end])

    KeyframeExporter.Export(output_prefix, database, lecture, idx_intervals, time_intervals, summary_indices,
                            summary_times, keyframes)

    KeyframeExporter.ExportGUIInfo(output_prefix, cc_times)

    return (summary_indices, summary_times, keyframes),

def main():
    # usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    in_files = [Parameters.Output_ST3D, Parameters.Output_Vid_Segment]
    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], in_files, Parameters.Output_Ext_Keyframes)

    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")

if __name__ == "__main__":
    main()
