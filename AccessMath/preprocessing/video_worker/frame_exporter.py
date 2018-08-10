import cv2
import json

class FrameExporter:
    def __init__(self, export_dir, img_extension='png'):
        self.width = None
        self.height = None

        self.all_metadata = {}
        self.img_format = img_extension if img_extension in ['jpg', 'png'] else 'png'

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