
#============================================================================
# Preprocessing Model for ST3D indexing - V 1.5
#
# Kenny Davila
# - Created:  March 13, 2018
#
#============================================================================

import os
import sys

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.config.parameters import Parameters

from AccessMath.preprocessing.content.person_detection_processor import PersonDetectionProcessor

def process_input(process, input_data):
    frame_info = input_data[0]

    # TODO: video resolution is available at source, and should be passed here. Currently assuming size
    source_width = 1920
    source_height = 1080

    # extract info ...
    pd_processor = PersonDetectionProcessor.from_raw_info(frame_info)

    # TODO: future versions should use the sub-sampled speaker detection results
    final_keyframes_idxs = pd_processor.identify_keyframes(True, True)

    print("-> Loading current annotation ... ")
    # Load annotation
    # ... file ...
    annotation_suffix = process.database.name + "_" + process.current_lecture.title.lower()
    input_prefix = process.database.output_annotations + "/" + annotation_suffix
    input_main_file = input_prefix + ".xml"

    output_dir = process.database.output_annotations + "_ext"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_filename = output_dir + "/" + annotation_suffix + ".xml"

    pd_processor.add_speaker_to_annotations(input_main_file, output_filename, source_width, source_height,
                                            final_keyframes_idxs)

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    # TODO: read this from parameters:
    input_prefix = Parameters.Output_PersonDetection
    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], input_prefix, None)

    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")

if __name__ == "__main__":
    main()
