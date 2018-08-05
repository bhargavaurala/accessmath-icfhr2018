
import cv2
import numpy as np
from scipy.signal import medfilt
from AccessMath.preprocessing.content.MLBinarizer import MLBinarizer

class ROIDetection:
    def __init__(self, ml_binarizer, temporal_blur_K, bin_t, edge_min_t, edge_max_t, roi_detector):
        self.width = 0
        self.height = 0
        self.frame_count = 0

        self.ml_binarizer = ml_binarizer

        # some general parameters
        self.temporal_blur_K = temporal_blur_K
        self.bin_t = bin_t
        self.edge_min_t = edge_min_t
        self.edge_max_t = edge_max_t

        # use external detector for the final process...
        self.roi_detector = roi_detector

        # computed output
        self.all_gray = None
        self.all_bg_diffs = None

        self.std_dev_image = None
        self.gray_median_image = None
        self.high_conf_bg = None
        self.high_conf_fg = None

        # final product
        self.ROI_edges = None
        self.ROI_polygon_mask = None
        self.ROI_score = None

        self.debug_mode = False
        self.debug_no_ROI = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height
        self.frame_count = 0

        self.all_gray = []
        self.all_bg_diffs = []
        self.std_dev_image = None
        self.gray_median_image = None
        self.high_conf_bg = None
        self.high_conf_fg = None

        self.ROI_edges = None
        self.ROI_polygon_mask = None
        self.ROI_score = None

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        self.frame_count += 1

        # convert to gray-scale ...
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.all_gray.append(img_gray)

        bg_diff = self.ml_binarizer.preprocessing(frame)

        self.all_bg_diffs.append(bg_diff)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(img_gray)

    def debug_frame(self, binary):
        pass

    def getWorkName(self):
        return "Region Of Interest Detector"

    def finalize(self):
        # convert to arrays, and remove original references ...
        full_gray = np.array(self.all_gray)
        self.all_gray = None

        full_bg_array = np.array(self.all_bg_diffs)
        self.all_bg_diffs = None

        # Apply a temporal median filter over the median subtracted frames ...
        # this removes most edges generated by the speaker and produces a cleaner
        # temporal STDev image for high confidence estimation of foreground pixels ...
        full_bg_array = medfilt(full_bg_array, [self.temporal_blur_K, 1, 1])

        # handwritting is stronger in the image of the
        # standard deviation of the differences from the median background estimations
        self.std_dev_image = np.std(full_bg_array, axis=0).astype(np.uint8)
        # Background is typically more visible than handwriting on the gray scale median
        self.gray_median_image = np.median(full_gray, axis=0).astype(np.uint8)

        if self.debug_no_ROI:
            # early termination
            return

        # detect edges on gray scale median, most likely belong to background objects
        self.high_conf_bg = cv2.Canny(self.gray_median_image, self.edge_min_t, self.edge_max_t)
        # binarize the STDev image of median subtracted edges, to obtain most likely handwriting pixels
        self.high_conf_fg = (self.std_dev_image >= self.bin_t).astype(np.uint8) * 255
        # otsu_t, self.high_conf_fg = cv2.threshold(self.std_dev_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # make them mutually exclusive by subtracting high conf. bg from high conf. fg.
        self.high_conf_fg[self.high_conf_bg > 0] = 0

        # create an image of input gray median image on 3 channels for showing elements in color
        video_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        video_image[:, :, 0] = self.gray_median_image.copy()
        video_image[:, :, 1] = self.gray_median_image.copy()
        video_image[:, :, 2] = self.gray_median_image.copy()

        box_info, debug_images = self.roi_detector.get_ROI(video_image, self.high_conf_bg, self.high_conf_fg, False, True)
        self.ROI_edges, self.ROI_polygon_mask, self.ROI_score = box_info
        line_image, ROI_image = debug_images

        if self.debug_mode:
            cv2.imwrite("ROI_DET_" + self.debug_video_name + "_GRAY_MEDIAN.png", self.gray_median_image.astype(np.uint8))
            cv2.imwrite("ROI_DET_" + self.debug_video_name + "_BG_STD.png", self.std_dev_image.astype(np.uint8))
            cv2.imwrite("ROI_DET_" + self.debug_video_name + "_LINES.png", line_image)
            cv2.imwrite("ROI_DET_" + self.debug_video_name + "_ROI.png", ROI_image)

