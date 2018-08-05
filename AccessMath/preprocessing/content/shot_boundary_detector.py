
import math
import cv, cv2
import numpy as np

class ShotBoundaryDetector:
    MethodHierarchicalClustering = 1
    FeaturesLuminosity = 1
    FeaturesColorHistChannel = 2
    FeaturesColorHistCombined = 3
    DistanceEuclidean = 1
    DistanceHistogramIntersection = 2

    def __init__(self, method_id, feature_type, distance):
        self.method_id = method_id
        self.feature_type = feature_type
        self.distance = distance

        self.lum_features = 1024
        self.clustering_threshold = 1200

        self.color_hist_bins = 8

        self.all_features = []

        self.all_frames = []

        self.shot_boundaries = None

    def get_luminosity_features(self, frame):
        gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY).astype('uint8')

        height, width = gray_scale.shape

        aspect_ratio = float(width) / float(height)

        self.small_h = int(math.sqrt(self.lum_features / aspect_ratio))
        self.small_w = int(aspect_ratio * self.small_h)

        small = cv2.resize(gray_scale, (self.small_w, self.small_h)).astype('float32')

        return small

    def get_color_histograms_by_channel(self, frame):
        bin_size = 256.0 / self.color_hist_bins
        bins = [x * bin_size for x in range(self.color_hist_bins + 1)]

        all_hist = []
        for channel in range(frame.shape[2]):
            hist, edges = np.histogram(frame[:, :, channel], bins)

            all_hist.append(hist)

        return np.hstack(all_hist)

    def get_color_histograms_combined(self, frame):
        vals = frame.reshape((frame.shape[0] * frame.shape[1], frame.shape[2]))

        bin_size = 256.0 / self.color_hist_bins
        bins = [x * bin_size for x in range(self.color_hist_bins + 1)]

        hist, edges = np.histogramdd(vals, bins)

        return hist.reshape((hist.shape[0] * hist.shape[1] * hist.shape[2]))

    def add_frame(self, frame):
        if self.feature_type == ShotBoundaryDetector.FeaturesLuminosity:
            features = self.get_luminosity_features(frame)
        elif self.feature_type == ShotBoundaryDetector.FeaturesColorHistChannel:
            features = self.get_color_histograms_by_channel(frame)
        elif self.feature_type == ShotBoundaryDetector.FeaturesColorHistCombined:
            features = self.get_color_histograms_combined(frame)
        else:
            features = None

        self.all_features.append(features)
        self.all_frames.append(frame)

    def euclidean_distance(self, feat1, feat2):
        return math.sqrt(np.power(feat1 - feat2, 2).sum())

    def histogram_distance(self, feat1, feat2):
        match = np.minimum(feat1, feat2).sum()

        p = feat1.sum()
        q = feat2.sum()

        # returns a symmetric value between 0.0 and 1.0 based on how much the histograms overlap
        # 0.0 - Complete overlap
        # 1.0 - No overlap at all
        return 1.0 - ((2.0 * match) / float(p + q))

    def get_distance(self, feat1, feat2):
        if self.distance == ShotBoundaryDetector.DistanceEuclidean:
            return self.euclidean_distance(feat1, feat2)
        elif self.distance == ShotBoundaryDetector.DistanceHistogramIntersection:
            return self.histogram_distance(feat1, feat2)
        else:
            # undefined distance
            return 0.0

    def boundaries_from_hierarchical_clustering(self):
        # start with all frames as individual clusters
        clusters = []
        for frame_idx in range(len(self.all_features)):
            feat = self.all_features[frame_idx]
            clusters.append((feat, frame_idx, frame_idx))

        # difference between clusters ... \
        differences = []
        for frame_idx in range(len(self.all_features) - 1):
            distance = self.get_distance(clusters[frame_idx][0], clusters[frame_idx + 1][0])

            # difference to the next cluster
            differences.append(distance)

        # now cluster until difference threshold is reached ....
        small_found = True
        while len(differences) > 0 and small_found:
            small_found = False
            small_offset = None
            #find the smallest ...
            for cluster_idx in range(len(differences)):
                if (differences[cluster_idx] < self.clustering_threshold and
                    (small_offset is None or differences[cluster_idx] < differences[small_offset])):
                    small_found = True
                    small_offset = cluster_idx

            #print(str((len(clusters), len(differences), small_offset)))

            if small_found:
                # merge clusters
                # compute combined features ...
                feats1, start1, end1 = clusters[small_offset]
                feats2, start2, end2 = clusters[small_offset + 1]
                # ... compute weights
                frames1 = end1 - start1 + 1
                frames2 = end2 - start2 + 1
                w1 = frames1 / float(frames1 + frames2)
                w2 = frames2 / float(frames1 + frames2)

                c_feats = feats1 * w1 + feats2 * w2

                # replace cluster at offset, remove next cluster
                del clusters[small_offset + 1]
                clusters[small_offset] = (c_feats, start1, end2)
                #print("now ..." + str(start1) + ", " + str(end2))

                # update differences
                if small_offset > 0:
                    # update previous
                    differences[small_offset - 1] = self.get_distance(clusters[small_offset - 1][0], clusters[small_offset][0])
                if small_offset + 1 < len(clusters):
                    # update current...
                    differences[small_offset] = self.get_distance(clusters[small_offset][0], clusters[small_offset + 1][0])
                    # delete next ....
                    del differences[small_offset + 1]
                else:
                    # current became the last one, delete difference
                    del differences[small_offset]


        print("Final clusters")
        for cluster_idx, (feats, start, end) in enumerate(clusters):
            print str((cluster_idx, start, end)),
            cv2.imwrite("output/images/cluster_middle_" + str(cluster_idx) + ".png", self.all_frames[int((start + end) / 2)])
            #cv2.imwrite("output/images/cluster_feats_" + str(cluster_idx) + ".png", feats)

        print(differences)


        print("Remmeber differences need to be relative to frame size (AVG difference or something)")

        self.shot_boundaries = None

    def finish_process(self):
        if self.method_id == ShotBoundaryDetector.MethodHierarchicalClustering:
            self.boundaries_from_hierarchical_clustering()
