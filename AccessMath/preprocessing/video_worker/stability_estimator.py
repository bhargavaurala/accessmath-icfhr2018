
import cv, cv2
from AccessMath.preprocessing.content.cc_stability_estimator import CCStabilityEstimator

class StabilityEstimator:
    def __init__(self, min_cc_stability):
        self.min_cc_stability = min_cc_stability

        self.estimator = None

        self.debug_mode = False
        self.debug_start = None
        self.debug_end = None
        self.debug_out_dir = None
        self.debug_video_name = None

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_title):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_title

    def initialize(self, width, height):
        self.estimator = CCStabilityEstimator(width, height, self.min_cc_stability)

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        self.estimator.verbose = self.debug_mode

        gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY)

        self.estimator.add_frame(gray_scale)

    def getWorkName(self):
        return "CC Stability Estimator"

    def finalize(self):
        self.estimator.finish_processing()

