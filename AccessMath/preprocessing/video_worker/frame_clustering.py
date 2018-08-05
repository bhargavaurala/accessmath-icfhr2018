
import cPickle
import math
import cv, cv2
import numpy as np
from sklearn.cluster import *

class FrameClustering:
    def __init__(self, n_clusters, max_features):
        self.width = 0
        self.height = 0

        self.frame_count = 0

        self.n_clusters = n_clusters
        self.max_features = max_features

        self.compressed_frames = None
        self.small_images = None

        self.small_w = None
        self.small_h = None

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frame_count = 0

        self.compressed_frames = []
        self.small_images = []

        aspect_ratio = float(width) / float(height)

        self.small_h = int(math.sqrt(self.max_features / aspect_ratio))
        self.small_w = int(aspect_ratio * self.small_h)

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name


    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        #print("Here: " + str(abs_time))

        self.frame_count += 1

        gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY).astype('uint8')

        self.compressed_frames.append(cv2.imencode(".png", gray_scale))

        small = cv2.resize(gray_scale, (self.small_w, self.small_h))

        features = small.reshape(self.small_w * self.small_h)
        self.small_images.append(features)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(small)

    def debug_frame(self, small):
        #cv2.imwrite(self.debug_out_dir + "/small_" + self.debug_video_name + "_" + str(self.frame_count) + ".png", small)
        pass

    def getWorkName(self):
        return "Frame Clustering"

    def finalize(self):
        print("Now Clustering")

        input_data = np.mat(self.small_images).astype(np.float32)

        """
        tempo_out = open("output/images/a_clust_input_" + self.debug_video_name + ".dat", "wb")
        cPickle.dump(input_data, tempo_out, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.compressed_frames, tempo_out, cPickle.HIGHEST_PROTOCOL)
        tempo_out.close()
        """

        #clustering = KMeans(self.n_clusters)
        clustering = AgglomerativeClustering(n_clusters=8)
        #clustering = DBSCAN(eps=1000.0, min_samples=25) #500-1000

        clusters = clustering.fit_predict(input_data)

        # for now save results
        n_points = len(self.small_images)
        cluster_sizes = [0 for x in range(max(clusters) + 1)]
        noise_size = 0
        for idx in range(n_points):
            flag, raw_data = self.compressed_frames[idx]
            img = cv2.imdecode(raw_data, cv.CV_LOAD_IMAGE_GRAYSCALE)

            if clusters[idx] >= 0:
                c_str = str(clusters[idx])
                cluster_sizes[clusters[idx]] += 1
            else:
                c_str = "noise"
                noise_size += 1

            fname = (self.debug_out_dir + "/clust_" + self.debug_video_name +
                     "_cls_" + c_str + "_" + str(idx) + ".png")

            cv2.imwrite(fname, img)


        print("Cluster sizes:")
        print(cluster_sizes)
        print("Noise: " + str(noise_size))