
import os
import sys

from AccessMath.preprocessing.video_worker.frame_exporter import FrameExporter
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.config.parameters import Parameters

def create_VOC_dirs(lecture_title):
    lecture_dir = Parameters.Output_FrameExport + lecture_title + '/'
    voc_subdirs = ['Annotations', 'ImageSets', 'JPEGImages']
    for voc_subdir in voc_subdirs:
        export_path = lecture_dir + voc_subdir
        if not os.path.isdir(export_path):
            os.makedirs(export_path)

def get_worker(process):
    create_VOC_dirs(process.current_lecture.title)
    export_dir = Parameters.Output_FrameExport + process.current_lecture.title + "/JPEGImages"
    frame_exporter = FrameExporter(export_dir, img_extension=Parameters.Output_FrameExport_ImgExtension)
    return frame_exporter

def main():
    # usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, None)

    if not process.initialize():
        return

    fps = 1.0 # 1.0
    process.start_video_processing(fps, get_worker, None, 0, True)

    print("Finished")


if __name__ == "__main__":
    main()

