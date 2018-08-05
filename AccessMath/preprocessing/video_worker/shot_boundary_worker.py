import cv2

from AccessMath.preprocessing.content.shot_boundary_detector import ShotBoundaryDetector

class ShotBoundaryWorker:
    def __init__(self, method_id, feature_type, distance):
        self.frame_count = 0
        self.sb_detector = None
        self.method_id = method_id
        self.feature_type = feature_type
        self.distance = distance

        self.lum_features = None
        self.color_hist_bins = None
        self.clustering_threshold = None

        self.all_times = []

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.frame_count = 0
        self.sb_detector = ShotBoundaryDetector(self.method_id, self.feature_type, self.distance)

        if self.method_id == ShotBoundaryDetector.MethodHierarchicalClustering:
            self.sb_detector.clustering_threshold = self.clustering_threshold
        if self.feature_type == ShotBoundaryDetector.FeaturesLuminosity:
            self.sb_detector.lum_features = self.lum_features
        elif (self.feature_type == ShotBoundaryDetector.FeaturesColorHistChannel or
            self.feature_type == ShotBoundaryDetector.FeaturesColorHistCombined):
            self.sb_detector.color_hist_bins = self.color_hist_bins


    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name


    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        self.frame_count += 1

        self.sb_detector.add_frame(frame)
        self.all_times.append(abs_time)


        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(frame)

    def debug_frame(self, frame):
        out_name = self.debug_out_dir + "/binary_" + self.debug_video_name + "_" + str(self.frame_count) + ".png"
        cv2.imwrite(out_name, frame)


    def getWorkName(self):
        return "Shot Boundary Detection"

    def finalize(self):
        print("Detecting the boundaries")
        self.sb_detector.finish_process()
