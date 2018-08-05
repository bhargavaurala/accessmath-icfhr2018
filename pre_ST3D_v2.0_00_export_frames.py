
import os
import sys
import cv2
import json

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.config.parameters import Parameters

class FrameExporter:
    def __init__(self, export_dir, img_format='png'):
        self.width = None
        self.height = None

        self.all_metadata = {}
        self.img_format = img_format if img_format in ['jpg', 'png'] else 'png'

        # directory where results will be stored ...
        self.export_dir = export_dir

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.all_metadata = {}

    def getWorkName(self):
        return "Raw Frame Exporter"

    def handleFrame(self, frame, last_frame, video_idx, frame_time, current_time, frame_idx):
        # Compute and export sample frame metadata
        self.all_metadata[frame_idx] = {
            "frame_idx": frame_idx,
            "abs_time": frame_time,
            "video_idx": video_idx,
            "video_time": current_time,
            "width": frame.shape[1],
            "height": frame.shape[0]
        }

        # Output file names ...
        out_img_filename = "{0:s}/{1:d}.{2:s}".format(self.export_dir, frame_idx, self.img_format)
        # ... save image ...
        if self.img_format == 'png':
            cv2.imwrite(out_img_filename, frame)
        else:
            cv2.imwrite(out_img_filename, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))

    def finalize(self):
        out_json_filename = "{0:s}/index.json".format(self.export_dir)

        with open(out_json_filename, "w") as out_file:
            json.dump(self.all_metadata, out_file)

        print("-> Metadata saved to: {0:s}".format(out_json_filename))

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
    frame_exporter = FrameExporter(export_dir)
    return frame_exporter

def get_results():
    return None

def main():
    # usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, None)

    if not process.initialize():
        return

    fps = 1.0 # 1.0
    process.start_video_processing(fps, get_worker, get_results, 0, True)

    print("Finished")


if __name__ == "__main__":
    main()

