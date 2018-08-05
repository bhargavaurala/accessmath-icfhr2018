
import cv2
from AccessMath.preprocessing.content.binarizer import Binarizer

class FrameBinarizer:
    def __init__(self):
        self.width = 0
        self.height = 0

        self.frame_count = 0
        # some general parameters
        self.binarization_method = Binarizer.MethodEdgeBased

        # shared by multiple methods
        self.sigma_color = None
        self.sigma_space = None
        self.bluring_ksize = None

        # for BG subtraction v1.0 and v 1.1
        self.threshold = None
        self.min_pixels = None

        # for BG subtraction v1.0
        self.disk_size = None
        # for BG subtraction v1.1
        self.dark_background = False


        self.last_binary = None

        self.frame_times = None
        self.frame_indices = None
        self.compressed_frames = None

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def set_bg_subtraction_method(self, bluring_ksize=5, disk_size=14, threshold=0.93, min_pixels=8):
        self.binarization_method = Binarizer.MethodBackgroundSubstraction

        self.bluring_ksize = bluring_ksize
        self.disk_size = disk_size
        self.threshold = threshold
        self.min_pixels = min_pixels

    def set_bg_subtraction_K(self, dark_background=False, sigma_color=4.0, sigma_space=4.0, bluring_ksize=33,
                             threshold=10, min_pixels=5):

        self.binarization_method = Binarizer.MethodSubtractionK

        self.dark_background = dark_background

        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.bluring_ksize = bluring_ksize

        self.threshold = threshold
        self.min_pixels = min_pixels

    def set_chalkboard_method(self, bluring_ksize=5, disk_size=14, threshold=25, min_pixels=8):
        self.binarization_method = Binarizer.MethodChalkboard

        self.bluring_ksize = bluring_ksize
        self.disk_size = disk_size
        self.threshold = threshold
        self.min_pixels = min_pixels

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frame_count = 0

        self.frame_times = []
        self.frame_indices = []
        self.compressed_frames = []

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        self.frame_count += 1

        if self.binarization_method == Binarizer.MethodSubtractionK:
            binary = Binarizer.bgSubtractionKBinarization(frame, self.dark_background, self.sigma_color, self.sigma_space,
                                                          self.bluring_ksize, self.threshold, self.min_pixels)
        elif self.binarization_method == Binarizer.MethodChalkboard:
            binary = Binarizer.chalkboardBinarization(frame, self.bluring_ksize, self.disk_size, self.threshold,
                                                      self.min_pixels)
        elif self.binarization_method == Binarizer.MethodBackgroundSubstraction:
            binary = Binarizer.backgroundSubtractionBinarization(frame, self.bluring_ksize, self.disk_size,
                                                                 self.threshold, self.min_pixels)
        else:
            # by default use the edge based method
            binary = Binarizer.edgeBasedRegionBinarization(frame)

        flag, raw_data = cv2.imencode(".png", binary)
        self.last_binary = binary

        self.compressed_frames.append(raw_data)
        self.frame_indices.append(abs_frame_idx)
        self.frame_times.append(abs_time)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(binary)

    def debug_frame(self, binary):
        out_name = self.debug_out_dir + "/binary_" + self.debug_video_name + "_" + str(self.frame_count) + ".png"
        cv2.imwrite(out_name, binary)


    def getWorkName(self):
        return "Frame Binarizer"

    def finalize(self):
        pass