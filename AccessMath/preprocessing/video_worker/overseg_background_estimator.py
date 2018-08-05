
import cPickle
import cv, cv2
import numpy as np
import math
import random

from AccessMath.preprocessing.content.cc_stability_estimator import CCStabilityEstimator
from AccessMath.preprocessing.tools.image_clustering import ImageClustering

#from scipy import stats
#import cPickle

#===================================================================
# Routines for simple overlapped segment analysis based on basic
# statistics obtained from the block
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2015
#
# Modified by:
#   - Kenny Davila (Sept, 2015)
#
#===================================================================

class OverSegBackgroundEstimator:
    CriterionRegionMedian = 0
    CriterionCCStability = 1

    def __init__(self, overlap_size, frames_segment):
        self.width = None
        self.height = None
        self.segments = None

        self.overlap_size = overlap_size
        self.frames_segment = frames_segment

        self.block_images = None
        self.last_start = None

        self.sorting_criterion = OverSegBackgroundEstimator.CriterionRegionMedian

        self.clusters_max_n = 1
        self.clusters_max_std_dev = 1.0
        self.clusters_min_best_images = 3
        self.cluster_features = 64
        self.cluster_w = None
        self.cluster_h = None
        self.block_features = None

        self.grid_cols = None
        self.grid_rows = None
        self.grid_best_prc = 1.0
        self.grid_col_lims = None
        self.grid_row_lims = None
        self.grid_col_boundaries = None
        self.grid_row_boundaries = None
        self.grid_masks = None

        self.cc_stable_frames = None
        self.cc_stable_score = None
        self.cc_stable_background = None

        self.debug_mode = False
        self.debug_out_dir = None
        self.debug_video_name = ""

    def set_clustering(self, max_clusters=2, max_std_dev=15.0, max_features=1024, min_best_images=3):
        self.clusters_max_n = max_clusters
        self.clusters_max_std_dev = max_std_dev
        self.cluster_features=max_features
        self.clusters_min_best_images = min_best_images

    def set_grid(self, grid_rows=4, grid_cols=8, grid_best_prc=0.1):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_best_prc = grid_best_prc

    def set_cc_stability(self, use_stability, min_stable_frames=3, min_stable_score=0.8, min_stable_background=0.5):
        if use_stability:
            self.sorting_criterion = OverSegBackgroundEstimator.CriterionCCStability

            self.cc_stable_frames = min_stable_frames
            self.cc_stable_score = min_stable_score
            self.cc_stable_background = min_stable_background
        else:
            self.sorting_criterion = OverSegBackgroundEstimator.CriterionRegionMedian

    def compute_overlapped_boundaries(self, cell_boundaries):
        count_cells = len(cell_boundaries) - 1

        boundaries = []
        for cell in range(count_cells):
            if cell > 0:
                start_cell = (cell_boundaries[cell] + cell_boundaries[cell - 1]) / 2
            else:
                start_cell = cell_boundaries[cell]
            if cell + 1 < count_cells:
                end_cell = (cell_boundaries[cell + 1] + cell_boundaries[cell + 2]) / 2
            else:
                end_cell = cell_boundaries[cell + 1]

            boundaries.append((start_cell, end_cell))

        return boundaries

    def generate_bilinear_interpolation_mask(self, row, col, total_rows, total_cols, height, width, center_y, center_x):
        horizontal_mask = np.ones((height, width), dtype=np.float32)
        vertical_mask = np.ones((height, width), dtype=np.float32)

        # generate horizontal weights
        if col > 0:
            # left side ...
            for tempo_col in range(center_x):
                horizontal_mask[:, tempo_col] = tempo_col / float(center_x)

        if col + 1 < total_cols:
            # right size ...
            for tempo_col in range(center_x + 1, width):
                horizontal_mask[:, tempo_col] = 1.0 - (tempo_col - center_x) / float(width - center_x)

        # generate vertical weights ...
        if row > 0:
            # left side ...
            for tempo_row in range(center_y):
                vertical_mask[tempo_row, :] = tempo_row / float(center_y)

        if row + 1 < total_rows:
            # right size ...
            for tempo_row in range(center_y + 1, height):
                vertical_mask[tempo_row, :] = 1.0 - (tempo_row - center_y) / float(height - center_y)

        return np.multiply(horizontal_mask, vertical_mask)

    def initialize(self, width, height):
        self.width = width
        self.height = height
        self.segments = []

        aspect_ratio = float(width) / float(height)

        self.cluster_h = int(math.sqrt(self.cluster_features / aspect_ratio))
        self.cluster_w = int(aspect_ratio * self.cluster_h)

        self.block_features = np.zeros((self.frames_segment * self.overlap_size, self.cluster_h * self.cluster_w), dtype=np.float32)

        self.grid_row_lims = [int(height / self.grid_rows) * idx + min(idx, height % self.grid_rows) for idx in range(self.grid_rows + 1)]
        self.grid_col_lims = [int(width / self.grid_cols) * idx + min(idx, width % self.grid_cols) for idx in range(self.grid_cols + 1)]

        self.grid_row_boundaries = self.compute_overlapped_boundaries(self.grid_row_lims)
        self.grid_col_boundaries = self.compute_overlapped_boundaries(self.grid_col_lims)

        self.grid_masks = []
        for row in range(self.grid_rows):
            self.grid_masks.append([])

            start_y, end_y = self.grid_row_boundaries[row]

            for col in range(self.grid_cols):
                start_x, end_x = self.grid_col_boundaries[col]

                mask = self.generate_bilinear_interpolation_mask(row, col, self.grid_rows, self.grid_cols,
                                                                 end_y - start_y, end_x - start_x,
                                                                 (self.grid_row_lims[row] + self.grid_row_lims[row + 1]) / 2 - start_y,
                                                                 (self.grid_col_lims[col] + self.grid_col_lims[col + 1]) / 2 - start_x)

                self.grid_masks[row].append(mask)

        self.block_images = np.zeros((self.frames_segment * self.overlap_size, height, width), dtype=np.uint8)
        self.last_start = []

    def median_from_grid(self):
        # initialization
        n_points, height, width = self.block_images.shape


        if self.sorting_criterion == OverSegBackgroundEstimator.CriterionCCStability:
            # compute CC stability per frame ...
            cc_data = CCStabilityEstimator.find_block_stable_cc(self.block_images, self.cc_stable_score, self.cc_stable_background, True)
            cc_objects, frames_per_cc, ccs_per_frame, bg_mask = cc_data

        else:
            cc_objects, frames_per_cc, ccs_per_frame, bg_mask = (None, None, None, None)

        background_model = np.zeros((self.height, self.width), dtype=np.float32)

        region_diff = [[[] for x in range(self.grid_cols)] for y in range(self.grid_rows)]

        # Extract the overlapped patches with their scores
        for img_idx in range(n_points):
            img = self.block_images[img_idx, :, :]

            if self.sorting_criterion == OverSegBackgroundEstimator.CriterionRegionMedian:
                # local estimate based on local median
                median_lum = np.median(img, None)
                img_diff = np.abs(img.astype('int32') - median_lum)
            elif self.sorting_criterion == OverSegBackgroundEstimator.CriterionCCStability:
                # estimate using global information of CC stability
                img_diff = np.zeros((height, width), dtype=np.uint8)

                # for each cc in current frame
                for cc_idx, local_cc in ccs_per_frame[img_idx]:
                    # check if the CC is not stable (appears in few frames)
                    if len(frames_per_cc[cc_idx]) < self.cc_stable_frames:
                        # add unstable to noise image
                        img_diff[local_cc.min_y:local_cc.max_y +1, local_cc.min_x:local_cc.max_x +1] += local_cc.img

                # remove stable background pixels from the score ...
                img_diff[bg_mask] = 0

            else:
                raise Exception("Invalid Sorting Criterion for Grid Tiles")

            for row in range(self.grid_rows):
                start_y, end_y = self.grid_row_boundaries[row]
                for col in range(self.grid_cols):
                    start_x, end_x = self.grid_col_boundaries[col]

                    total_diff = img_diff[start_y:end_y, start_x:end_x].sum()

                    region_diff[row][col].append((total_diff, img[start_y:end_y, start_x:end_x]))

        for row in range(self.grid_rows):
            start_y, end_y = self.grid_row_boundaries[row]

            for col in range(self.grid_cols):
                start_x, end_x = self.grid_col_boundaries[col]

                mask = self.grid_masks[row][col]

                # sort ...
                ordered = sorted(region_diff[row][col], key=lambda x:x[0])

                patch_height = ordered[0][1].shape[0]
                patch_width = ordered[0][1].shape[1]
                block_data = np.zeros((n_points, patch_height, patch_width))
                for idx in range(len(region_diff[row][col])):
                    block_data[idx, :, :] = ordered[idx][1]

                if self.clusters_max_n > 1:
                    cluster_indices = ImageClustering.cluster_images(block_data, self.cluster_features, self.clusters_max_n, self.clusters_max_std_dev)

                    # now, compute the score for each cluster ...
                    cluster_scores = []
                    for cluster_idx in range(len(cluster_indices)):
                        total_score = 0.0
                        for img_idx in cluster_indices[cluster_idx]:
                            total_score += ordered[img_idx][0]

                        total_score /= (len(cluster_indices[cluster_idx]) * patch_width * patch_height)

                        cluster_scores.append((total_score, cluster_idx))
                        #print("Cluster: " + str(cluster_idx) + ", final score: " + str(total_score))


                    cluster_scores = sorted(cluster_scores)
                    top_clusters = 0
                    best_count = 0
                    while best_count < self.clusters_min_best_images:
                        current_size = len(cluster_indices[cluster_scores[top_clusters][1]])

                        top_clusters += 1
                        best_count += current_size

                    # Use best cluster(s)
                    cluster_data = np.zeros((best_count, patch_height, patch_width), dtype=np.uint8)
                    best_offset = 0
                    for best_cluster in range(top_clusters):
                        cluster_idx = cluster_scores[best_cluster][1]

                        for offset, img_idx in enumerate(cluster_indices[cluster_idx]):
                            cluster_data[best_offset + offset, :, :] = ordered[img_idx][1]

                        best_offset += len(cluster_indices[cluster_idx])

                    patch_median = np.median(cluster_data, axis=0)
                else:
                    best_k = int(self.grid_best_prc * n_points)
                    patch_median = np.median(block_data[0:best_k, :, :], axis=0)

                background_model[start_y:end_y, start_x:end_x] += np.multiply(patch_median, mask)

        return background_model

    def set_debug_mode(self, mode, out_dir, video_name):
        self.debug_mode = mode
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    def handleFrame(self, frame, last_frame, n_block, v_index, abs_time, rel_Time):
        raise Exception("Class OverlappedSegmentAnalyzer not designed for sequential processing")

    def compute_std(self, current_images):
        height = current_images.shape[1]
        width = current_images.shape[2]

        std_image = np.zeros((height, width), dtype=np.float64)

        square_side = 500
        start_y = 0
        while start_y < height:
            end_y = start_y + square_side

            start_x = 0
            while start_x < width:
                end_x = start_x + square_side

                std_image[start_y:end_y, start_x:end_x] = np.std(current_images[:, start_y:end_y,start_x:end_x], 0)

                start_x += square_side

            start_y += square_side

        return std_image



    def handleBlock(self, frames, n_block, start_abs, end_abs):
        if len(frames) == 0 or len(frames) != self.frames_segment:
            raise Exception("Unexpected number of frames received")

        #print(str(n_block) + "\t" + str(start_abs) + " - " + str(end_abs) + "\t" + str(len(frames)))

        # first, obtain the gray scale images ...
        height = frames[0].shape[0]
        width = frames[0].shape[1]

        # start of current segment
        self.last_start.append(start_abs)
        current_start = self.last_start[0]
        if len(self.last_start) == self.overlap_size:
            del self.last_start[0]

        # convert the newer images to grayscale and place in the temporal array
        # also get the feature version (smaller) for clustering
        offset = (n_block % self.overlap_size) * self.frames_segment

        for idx, frame in enumerate(frames):
            gray_scale = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
            self.block_images[offset + idx, :, :] = gray_scale

            small = cv2.resize(gray_scale, (self.cluster_w, self.cluster_h))
            self.block_features[offset + idx, :] = small.reshape(self.cluster_w * self.cluster_h)

        """
        if n_block in [8, 12, 25, 32, 38]:
            tempo_out = open("output/images/b_clust_input_" + self.debug_video_name + "_" + str(n_block) + ".dat", "wb")
            cPickle.dump(np.zeros((2,2)), tempo_out, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.block_images, tempo_out, cPickle.HIGHEST_PROTOCOL)
            tempo_out.close()
        """
        if n_block + 1 >= self.overlap_size:
            # get the uniform sample median
            median_image = np.median(self.block_images, 0)
            flag, compressed_uniform_median = cv2.imencode(".png", median_image)

            grid_median = self.median_from_grid()
            flag, compressed_grid_median = cv2.imencode(".png", grid_median)

            segment_info = {
                "n_block": n_block,
                "start_abs": current_start,
                "end_abs": end_abs,
                "uniform_background": compressed_uniform_median,
                "grid_background": compressed_grid_median,
            }

            self.segments.append(segment_info)

            # debug ...
            if self.debug_mode:
                self.debug_save(n_block, median_image, grid_median)


    def debug_save(self, n_block, uniform_median, other_median):
        common_prefix = self.debug_out_dir + "/" + self.debug_video_name + "_"
        common_sufix = "_" + str(n_block) + ".png"

        cv2.imwrite(common_prefix + "uni_background" + common_sufix, uniform_median)
        cv2.imwrite(common_prefix + "other_background" + common_sufix, other_median)

    def getWorkName(self):
        return "Overlapped Segment Analyzer"


    def finalize(self):
        pass

    def postProcess(self):
        pass