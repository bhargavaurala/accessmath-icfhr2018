
import cv2
import numpy as np
import time
from .binarizer import Binarizer
from .labeler import Labeler
from AccessMath.preprocessing.tools.interval_index import IntervalIndex


class CCStabilityEstimator:
    def __init__(self, width, height, min_recall, min_precision, max_gap, verbose=False):
        self.width = width
        self.height = height
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_gap = max_gap
        self.unique_cc_objects = []
        self.unique_cc_frames = []
        self.cc_idx_per_frame = []
        self.cc_int_index_x = IntervalIndex(True)
        self.cc_int_index_y = IntervalIndex(True)
        self.fake_age = np.zeros((height, width), dtype=np.float32)

        self.img_idx = 0
        self.tempo_count = 0

        # optimizing the matching ...
        self.cc_last_frame = []
        self.cc_active = []

        self.verbose = verbose

    def get_raw_cc_count(self):
        total = 0

        for current_frame in self.cc_idx_per_frame:
            total += len(current_frame)

        return total

    def add_frame(self, img, input_binary=False):
        # get the CC
        if input_binary:
            # use given binary
            binary = img
        else:
            # binarize
            binary = Binarizer.backgroundSubtractionBinarization(img.astype('uint8'))

        current_cc = Labeler.extractSpatioTemporalContent(binary, self.fake_age)
        current_cc_idxs = []
        if self.img_idx == 0:
            # simply copy all
            for cc in current_cc:
                self.unique_cc_objects.append(cc) # CC objet

                # frames on which the CC appears, raw label assigend to that CC
                self.unique_cc_frames.append([(0, cc.cc_id + 1)])

                cc_idx = len(self.unique_cc_objects) - 1
                current_cc_idxs.append((cc_idx, cc))

                self.cc_last_frame.append(0)
                self.cc_active.append(cc_idx)


                # add to indices ...
                self.cc_int_index_x.add(cc.min_x, cc.max_x + 1, cc_idx)
                self.cc_int_index_y.add(cc.min_y, cc.max_y + 1, cc_idx)

        else:
            # create indices for current CC
            other_index_x = IntervalIndex(True)
            other_index_y = IntervalIndex(True)
            for cc_idx, cc in enumerate(current_cc):
                other_index_x.add(cc.min_x, cc.max_x + 1, cc_idx)
                other_index_y.add(cc.min_y, cc.max_y + 1, cc_idx)

            # compute CC with matching regions
            set_x = set(other_index_x.find_matches(self.cc_int_index_x))
            set_y = set(other_index_y.find_matches(self.cc_int_index_y))

            # list of all pairs of CC with intersecting intervals in X and Y
            merged = sorted(list(set_x.intersection(set_y)))
            self.tempo_count += len(merged)

            # check every matching CC
            pre_add_size = len(self.cc_active)
            next_match_idx = 0
            for cc_idx, cc in enumerate(current_cc):
                found = False
                # check all matches in the list of matches for current CC
                while next_match_idx < len(merged) and merged[next_match_idx][0] == cc_idx:
                    if not found:
                        prev_idx = merged[next_match_idx][1]
                        prev_cc = self.unique_cc_objects[prev_idx]

                        recall, precision = cc.getOverlapFMeasure(prev_cc, False, False)
                        if recall >= self.min_recall and precision >= self.min_precision:
                            # assume they are equivalent
                            found = True
                            self.unique_cc_frames[prev_idx].append((self.img_idx, cc.cc_id + 1))
                            current_cc_idxs.append((prev_idx, cc))

                            # update last frame seen for this cc...
                            self.cc_last_frame[prev_idx] = self.img_idx

                    next_match_idx += 1

                # Not match was found?
                if not found:
                    # add
                    self.unique_cc_objects.append(cc)
                    self.unique_cc_frames.append([(self.img_idx, cc.cc_id + 1)])

                    new_cc_idx = len(self.unique_cc_objects) - 1
                    current_cc_idxs.append((new_cc_idx, cc))

                    self.cc_last_frame.append(self.img_idx)
                    self.cc_active.append(new_cc_idx)

                    # add to indices ...
                    self.cc_int_index_x.add(cc.min_x, cc.max_x + 1, new_cc_idx)
                    self.cc_int_index_y.add(cc.min_y, cc.max_y + 1, new_cc_idx)

            # remove CC that are no longer active
            pre_remove_size = len(self.cc_active)
            tempo_pos = 0
            while tempo_pos < len(self.cc_active):
                cc_idx = self.cc_active[tempo_pos]

                if self.img_idx - self.cc_last_frame[cc_idx] >= self.max_gap:
                    # no longer active ..
                    # delete from active list
                    del self.cc_active[tempo_pos]

                    # delete from interval indices
                    cc = self.unique_cc_objects[cc_idx]
                    self.cc_int_index_x.remove(cc.min_x, cc.max_x + 1, cc_idx)
                    self.cc_int_index_y.remove(cc.min_y, cc.max_y + 1, cc_idx)

                    #print self.cc_last_frame[cc_idx],
                else:
                    # still active
                    tempo_pos += 1

            #print(str(len(self.cc_active)) + ", (Added: " + str(pre_remove_size - pre_add_size) + ", Removed: " + str(pre_remove_size - len(self.cc_active)) + ")")


        self.cc_idx_per_frame.append(current_cc_idxs)

        self.img_idx += 1

        if self.verbose:
            print("[" + str(self.img_idx) + " (" +  str(len(current_cc)) + ", " + str(len(self.unique_cc_objects)) + ")]", end="\r")


    def finish_processing(self):
        if self.verbose:
            print(".")

        print("Erase this final count (after done): " + str(self.tempo_count))

        self.fake_age = None

    def rebuilt_binary_images(self):
        rebuilt_frames = []
        for frame_ccs in self.cc_idx_per_frame:
            binary = np.zeros((self.height, self.width), dtype=np.uint8)
            for idx_cc, local_cc in frame_ccs:
                binary[local_cc.min_y:local_cc.max_y +1, local_cc.min_x:local_cc.max_x + 1] += local_cc.img

            rebuilt_frames.append(binary)

        return rebuilt_frames

    def split_stable_cc_by_gaps(self, max_gap, stable_min_frames):
        splitted_count = 0
        n_original_objects = len(self.unique_cc_objects)
        for idx_cc in range(n_original_objects):
            current_frames = self.unique_cc_frames[idx_cc]
            n_local = len(current_frames)

            current_group = [current_frames[0]]
            valid_groups = [current_group]

            for frame_offset in range(1, n_local):
                curr_frame_idx = current_frames[frame_offset][0]
                prev_frame_idx = current_frames[frame_offset - 1][0]

                current_gap = curr_frame_idx - prev_frame_idx
                if current_gap > max_gap:
                    # not acceptable gap ..
                    current_group = [current_frames[frame_offset]]
                    valid_groups.append(current_group)
                else:
                    # acceptable
                    current_group.append(current_frames[frame_offset])

            if len(valid_groups) >= 2 and n_local >= stable_min_frames:
                # replace the frames in the original cc list to only the first group ..
                self.unique_cc_frames[idx_cc] = valid_groups[0]

                # for each group (new CC)
                for group_offset in range(1, len(valid_groups)):
                    new_idx_cc = len(self.unique_cc_objects)
                    # add another reference to the original CC
                    self.unique_cc_objects.append(self.unique_cc_objects[idx_cc])

                    # add CC frame reference ...
                    self.unique_cc_frames.append(valid_groups[group_offset])

                    # for each frame where the orgiianl appeared ..
                    for frame_idx, local_cc in valid_groups[group_offset]:
                        # find the CC on the frame ...
                        for offset, (local_cc_idx, local_cc) in enumerate(self.cc_idx_per_frame[frame_idx]):
                            if local_cc_idx == idx_cc:
                                # replace
                                self.cc_idx_per_frame[frame_idx][offset] = (new_idx_cc, local_cc)
                                break

                splitted_count += 1

        return splitted_count

    def get_stable_cc_idxs(self, min_stable_frames):
        stable_idxs = []
        for idx_cc in range(len(self.unique_cc_objects)):
            if len(self.unique_cc_frames[idx_cc]) >= min_stable_frames:
                stable_idxs.append(idx_cc)

        return stable_idxs

    def get_temporal_index(self):
        temporal_index = []
        for idxs_per_frame in self.cc_idx_per_frame:
            temporal_index.append([cc_idx for cc_idx, local_cc in idxs_per_frame])

        return temporal_index

    def compute_overlapping_stable_cc(self, stable_idxs, temporal_window):
        n_objects = len(self.unique_cc_objects)
        n_stable = len(stable_idxs)

        all_overlapping_cc = [[] for x in range(n_objects)]
        time_overlapping_cc = [[] for x in range(n_objects)]
        total_intersections = 0

        for offset1 in range(n_stable):
            # get first stable
            idx_cc1 = stable_idxs[offset1]
            cc1 = self.unique_cc_objects[idx_cc1]

            cc1_t_start = self.unique_cc_frames[idx_cc1][0][0]
            cc1_t_end = self.unique_cc_frames[idx_cc1][-1][0]

            print("Processing: " + str(offset1),end="\r")

            for offset2 in range(offset1 + 1, n_stable):
                idx_cc2 = stable_idxs[offset2]
                cc2 =  self.unique_cc_objects[idx_cc2]

                cc2_t_start = self.unique_cc_frames[idx_cc2][0][0]
                cc2_t_end = self.unique_cc_frames[idx_cc2][-1][0]

                # check intersection in space
                recall, precision = cc1.getOverlapFMeasure(cc2, False, False)
                if recall > 0.0 or precision > 0.0:
                    matched_pixels = int(cc1.size * recall)
                    all_overlapping_cc[idx_cc1].append((idx_cc2, matched_pixels, cc2.size, cc1.size))
                    all_overlapping_cc[idx_cc2].append((idx_cc1, matched_pixels, cc1.size, cc2.size))

                    # now, check intersection in time (considering time window)
                    if cc1_t_end + temporal_window >= cc2_t_start and cc2_t_end >= cc1_t_start - temporal_window:
                        # they have some intersection in pixels....
                        time_overlapping_cc[idx_cc1].append((idx_cc2, recall, precision))
                        time_overlapping_cc[idx_cc2].append((idx_cc1, precision, recall))

                        total_intersections += 1

        return time_overlapping_cc, total_intersections, all_overlapping_cc

    def compute_groups(self, stable_idxs, overlapping_cc):
        n_stable = len(stable_idxs)

        # Determine Groups of CC's that coexist in time and space and treat them as single units...
        cc_groups = []
        group_idx_per_cc = {}
        for offset1 in range(n_stable):
            idx_cc1 = stable_idxs[offset1]

            if idx_cc1 in group_idx_per_cc:
                # use the existing group
                group_idx = group_idx_per_cc[idx_cc1]
            else:
                # create new group
                group_idx = len(cc_groups)
                cc_groups.append([idx_cc1])
                group_idx_per_cc[idx_cc1] = group_idx

            # for every CC that occupies the same space ..
            for idx_cc2, recall, precision in overlapping_cc[idx_cc1]:
                # if it is not in the current group, add it ...
                if idx_cc2 not in group_idx_per_cc:
                    # add to current group
                    group_idx_per_cc[idx_cc2] = group_idx
                    # add to the current group
                    cc_groups[group_idx].append(idx_cc2)
                else:
                    other_group_idx = group_idx_per_cc[idx_cc2]
                    if other_group_idx != group_idx:
                        # different group? merge ..
                        #print("Merging groups")
                        # for each element in the other group
                        for other_idx_cc in cc_groups[other_group_idx]:
                            # link to the current group
                            group_idx_per_cc[other_idx_cc] = group_idx
                            # add to the current group
                            cc_groups[group_idx].append(other_idx_cc)

                        # leave the other group empty
                        cc_groups[other_group_idx] = []

        # clean up empty groups
        final_cc_groups = []
        final_group_idx_per_cc = {}

        for group in cc_groups:
            if len(group) > 0:
                new_group_idx = len(final_cc_groups)
                final_cc_groups.append(group)

                for idx_cc in group:
                    final_group_idx_per_cc[idx_cc] = new_group_idx

        return final_cc_groups, final_group_idx_per_cc

    def compute_groups_temporal_information(self, cc_groups):
        # compute temporal information for each group
        n_frames = len(self.cc_idx_per_frame)
        group_ages = {}
        groups_per_frame = [[] for frame_idx in range(n_frames)]

        for group_idx, group in enumerate(cc_groups):
            if len(group) == 0:
                continue

            current_ages = []
            for cc_idx in group:
                g_first = self.unique_cc_frames[cc_idx][0][0]
                g_last = self.unique_cc_frames[cc_idx][-1][0]

                # add first
                if g_first not in current_ages:
                    current_ages.append(g_first)

                if g_last not in current_ages:
                    current_ages.append(g_last)

            current_ages = sorted(current_ages)

            group_ages[group_idx] = current_ages

            for frame_idx in range(current_ages[0], min(current_ages[-1] + 1, n_frames)):
                groups_per_frame[frame_idx].append(group_idx)

        return group_ages, groups_per_frame

    def compute_conflicting_groups(self, stable_idxs, all_overlapping_cc, n_groups, group_idx_per_cc):
        n_stable = len(stable_idxs)

        # for each stable CC
        conflicts = {group_idx:{} for group_idx in range(n_groups)}
        for offset1 in range(n_stable):
            idx_cc1 = stable_idxs[offset1]

            # for every CC that occupies the same space (same group or not)..
            for idx_cc2, matched_pixels, size_cc2, size_cc1 in all_overlapping_cc[idx_cc1]:

                # consider each link only once
                if idx_cc1 < idx_cc2:
                    unmatched_pixels = size_cc1 + size_cc2 - matched_pixels * 2

                    # check if they are on different groups
                    group_idx1 = group_idx_per_cc[idx_cc1]
                    group_idx2 = group_idx_per_cc[idx_cc2]
                    if group_idx1 != group_idx2:
                        # conflict found, add the total of matched pixels in the conflict
                        if group_idx2 in conflicts[group_idx1]:
                            conflicts[group_idx1][group_idx2]["matched"] += matched_pixels
                            conflicts[group_idx1][group_idx2]["unmatched"] += unmatched_pixels
                        else:
                            conflicts[group_idx1][group_idx2] = {
                                "matched": matched_pixels,
                                "unmatched": unmatched_pixels
                            }

                        if group_idx1 in conflicts[group_idx2]:
                            conflicts[group_idx2][group_idx1]["matched"] += matched_pixels
                            conflicts[group_idx2][group_idx1]["unmatched"] += unmatched_pixels
                        else:
                            conflicts[group_idx2][group_idx1] = {
                                "matched": matched_pixels,
                                "unmatched": unmatched_pixels
                            }

        return conflicts

    def compute_group_images_from_raw_binary(self, cc_groups, group_ages, binary_frames, segment_threshold):
        group_images = {}
        group_boundaries = {}

        for group_idx, group in enumerate(cc_groups):
            if len(group) == 0:
                continue

            # first, compute the combined boundaries...
            cc_idx = group[0]
            g_min_x = self.unique_cc_objects[cc_idx].min_x
            g_max_x = self.unique_cc_objects[cc_idx].max_x
            g_min_y = self.unique_cc_objects[cc_idx].min_y
            g_max_y = self.unique_cc_objects[cc_idx].max_y

            for cc_idx in group:
                g_min_x = min(g_min_x, self.unique_cc_objects[cc_idx].min_x)
                g_max_x = max(g_max_x, self.unique_cc_objects[cc_idx].max_x)
                g_min_y = min(g_min_y, self.unique_cc_objects[cc_idx].min_y)
                g_max_y = max(g_max_y, self.unique_cc_objects[cc_idx].max_y)

            group_boundaries[group_idx] = (g_min_x, g_max_x, g_min_y, g_max_y)

            # size ...
            g_width = g_max_x - g_min_x + 1
            g_height = g_max_y - g_min_y + 1

            # compute the actual images using temporal information ..
            current_images = []
            current_ages = group_ages[group_idx]

            for t_segment in range(len(current_ages) - 1):
                t_start = current_ages[t_segment]
                t_end = current_ages[t_segment + 1]

                g_mask = np.zeros((g_height, g_width), dtype=np.int32)

                for cc_idx in group:
                    cc = self.unique_cc_objects[cc_idx]

                    cc_first = self.unique_cc_frames[cc_idx][0][0]
                    cc_last = self.unique_cc_frames[cc_idx][-1][0]

                    # check if cc existed in current time segment ...
                    if cc_first <= t_end and t_start <= cc_last:
                        # add to current image
                        offset_x = cc.min_x - g_min_x
                        offset_y = cc.min_y - g_min_y

                        g_mask[offset_y:offset_y + cc.getHeight(), offset_x:offset_x + cc.getWidth()] += (cc.img // 255)

                # now get the mask ...
                g_mask = (g_mask > 0).astype('uint8') * 255


                # use the mask to obtain the image of the group for the current segment of time
                segment_img = np.zeros((g_height, g_width), dtype=np.int32)
                for frame_idx in range(t_start, t_end + 1):
                    local_patch = np.bitwise_and(binary_frames[frame_idx][g_min_y:g_max_y + 1, g_min_x:g_max_x + 1], g_mask) // 255
                    segment_img += local_patch

                segment_img = (segment_img * 255) // segment_img.max()
                #segment_img[segment_img < min_pixel_val] = 0
                segment_img = (segment_img > segment_threshold).astype(dtype=np.uint8) * 255

                #g_img = (g_img * 255) / g_img.max()
                #cv2.imwrite("output/images/fs_a_group_" + str(group_idx) + "_" + str(t_start) + ".png", segment_img)
                current_images.append(segment_img)

            group_images[group_idx] = current_images

        return group_images, group_boundaries

    def compute_group_images(self, cc_groups, group_ages, segment_threshold):
        group_images = {}
        group_boundaries = {}

        for group_idx, group in enumerate(cc_groups):
            if len(group) == 0:
                continue

            # first, compute the combined boundaries...
            cc_idx = group[0]
            g_min_x = self.unique_cc_objects[cc_idx].min_x
            g_max_x = self.unique_cc_objects[cc_idx].max_x
            g_min_y = self.unique_cc_objects[cc_idx].min_y
            g_max_y = self.unique_cc_objects[cc_idx].max_y

            for cc_idx in group:
                g_min_x = min(g_min_x, self.unique_cc_objects[cc_idx].min_x)
                g_max_x = max(g_max_x, self.unique_cc_objects[cc_idx].max_x)
                g_min_y = min(g_min_y, self.unique_cc_objects[cc_idx].min_y)
                g_max_y = max(g_max_y, self.unique_cc_objects[cc_idx].max_y)

            group_boundaries[group_idx] = (g_min_x, g_max_x, g_min_y, g_max_y)

            # size ...
            g_width = g_max_x - g_min_x + 1
            g_height = g_max_y - g_min_y + 1

            # compute the actual images using temporal information ..
            current_images = []
            current_ages = group_ages[group_idx]

            for t_segment in range(len(current_ages) - 1):
                t_start = current_ages[t_segment]
                t_end = current_ages[t_segment + 1]

                g_mask = np.zeros((g_height, g_width), dtype=np.int32)

                # get the sum of CCs for all frames where they appear ...
                for cc_idx in group:
                    cc = self.unique_cc_objects[cc_idx]

                    # cc_first = self.unique_cc_frames[cc_idx][0][0]
                    # cc_last = self.unique_cc_frames[cc_idx][-1][0]

                    cc_frames = len([f_idx for f_idx, _ in self.unique_cc_frames[cc_idx] if t_start <= f_idx <= t_end])

                    if cc_frames > 0:
                        offset_x = cc.min_x - g_min_x
                        offset_y = cc.min_y - g_min_y

                        mask_cut = g_mask[offset_y:offset_y + cc.getHeight(), offset_x:offset_x + cc.getWidth()]

                        # all CC pixels addes as many times as frames the entire CC appears
                        mask_cut += (cc.img // 255) * cc_frames

                segment_img = ((g_mask.astype(np.float64) / g_mask.max()) >= segment_threshold).astype(np.uint8) * 255

                current_images.append(segment_img)

            group_images[group_idx] = current_images

        return group_images, group_boundaries

    def frames_from_groups(self, cc_groups, group_boundaries, groups_per_frame, group_ages, group_images,
                           save_prefix=None, stable_min_frames=3, show_unstable=True):

        group_next_segment = [0 for group_idx in range(len(cc_groups))]
        clean_binary = []

        for img_idx, groups_in_frame in enumerate(groups_per_frame):
            reconstructed = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for group_idx in groups_in_frame:
                # find corresponding image ....
                current_ages = group_ages[group_idx]
                while current_ages[group_next_segment[group_idx] + 1] < img_idx:
                    # move to the next segment
                    group_next_segment[group_idx] += 1

                # use image of selected segment ...
                segment_img = group_images[group_idx][group_next_segment[group_idx]]

                # add to the image
                g_min_x, g_max_x, g_min_y, g_max_y = group_boundaries[group_idx]

                reconstructed[g_min_y:g_max_y + 1, g_min_x:g_max_x + 1, 0] += segment_img
                reconstructed[g_min_y:g_max_y + 1, g_min_x:g_max_x + 1, 1] += segment_img
                if not show_unstable:
                    reconstructed[g_min_y:g_max_y + 1, g_min_x:g_max_x + 1, 2] += segment_img


            if show_unstable:
                for cc_idx, local_cc in self.cc_idx_per_frame[img_idx]:
                    if len(self.unique_cc_frames[cc_idx]) < stable_min_frames:
                        # add unstable in the red channel
                        cc = local_cc
                        reconstructed[cc.min_y:cc.max_y +1, cc.min_x:cc.max_x + 1, 2] += cc.img

            #print(group_next_segment)
            if not save_prefix is None:
                cv2.imwrite(save_prefix + "_stab_" + str(img_idx) + ".png", reconstructed)
                cv2.imwrite(save_prefix + "_clean_" + str(img_idx) + ".png", reconstructed[:,:, 0])

            flag, raw_data = cv2.imencode(".png", reconstructed[:,:, 0])
            clean_binary.append(raw_data)

        return clean_binary

    @staticmethod
    def find_block_stable_cc(block_images, min_f_score, min_stable_bg = 0.50, verbose=False):
        n_points, height, width = block_images.shape

        estimator = CCStabilityEstimator(width, height, min_f_score, min_stable_bg, verbose)

        for img_idx in range(n_points):
            img = block_images[img_idx, :, :]

            estimator.add_frame(img)

        return estimator.unique_cc_objects, estimator.unique_cc_frames, estimator.cc_idx_per_frame, estimator.stable_background_mask

    @staticmethod
    def compute_overlapping_CC_groups(cc_objects):
        n_objects = len(cc_objects)

        all_overlapping_cc = [[x] for x in range(n_objects)]

        # compute all pairwise overlaps
        for idx1 in range(n_objects):
            # get first stable
            cc1 = cc_objects[idx1]

            for idx2 in range(idx1 + 1, n_objects):
                cc2 = cc_objects[idx2]

                # check intersection in space
                recall, precision = cc1.getOverlapFMeasure(cc2, False, False)
                if recall > 0.0 or precision > 0.0:
                    all_overlapping_cc[idx1].append(idx2)
                    all_overlapping_cc[idx2].append(idx1)

        # find groups containing pairs of overlapping objects (transitive overlap)
        group_overlap_idx = [x for x in range(n_objects)]
        merged_groups = {x: {x} for x in range(n_objects)}
        for idx in range(n_objects):
            # check the group of the current object
            merged_idx1 = group_overlap_idx[idx]

            current_group = all_overlapping_cc[idx]
            for other_idx in current_group[1:]:
                # check the group of an overlapping obejct
                merged_idx2 = group_overlap_idx[other_idx]

                # if it is different, then merge!
                if merged_idx1 != merged_idx2:
                    # 1 )create a single larger group
                    merged_groups[merged_idx1] = merged_groups[merged_idx1].union(merged_groups[merged_idx2])
                    # 2 ) Redirect each element on the group that will disappear, to the newer larger group
                    for old_group_idx in merged_groups[merged_idx2]:
                        group_overlap_idx[old_group_idx] = merged_idx1
                    # 3) Delete old unreferenced group
                    del merged_groups[merged_idx2]

        # finally, split by overlapping and no overlapping ..
        overlapping_groups = []
        no_overlaps = []
        for group_idx in merged_groups:
            merged_list = list(merged_groups[group_idx])
            if len(merged_list) == 1:
                no_overlaps.append(merged_list[0])
            else:
                overlapping_groups.append(merged_list)

        return overlapping_groups, no_overlaps