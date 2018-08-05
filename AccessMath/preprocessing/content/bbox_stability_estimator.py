
import cv2
import numpy as np

from AccessMath.preprocessing.tools.interval_index import IntervalIndex


class BBoxStabilityEstimator:
    RefinePerBoxPerSegment = 1
    RefineAllBoxesPerSegment = 2
    RefineAllBoxesAllSegments = 3

    def __init__(self, width, height, min_combined_ratio, min_temporal_IOU, max_gap, verbose=False):
        self.width = width
        self.height = height
        self.min_combined_ratio = min_combined_ratio
        self.min_temporal_IOU = min_temporal_IOU
        self.max_gap = max_gap

        self.unique_bbox_objects = []
        self.unique_bbox_frames = []
        self.bbox_idx_per_frame = []
        self.bbox_int_index_x = IntervalIndex(True)
        self.bbox_int_index_y = IntervalIndex(True)

        self.img_idx = 0
        self.tempo_count = 0

        # optimizing the matching ...
        self.bbox_last_frame = []
        self.bbox_active = []

        self.verbose = verbose

    def get_raw_bbox_count(self):
        total = 0

        for current_frame in self.bbox_idx_per_frame:
            total += len(current_frame)

        return total

    def get_inter_bbox(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        comb_inter_x1 = max(b1_x1, b2_x1)
        comb_inter_y1 = max(b1_y1, b2_y1)
        comb_inter_x2 = min(b1_x2, b2_x2)
        comb_inter_y2 = min(b1_y2, b2_y2)

        return comb_inter_x1, comb_inter_y1, comb_inter_x2, comb_inter_y2

    def get_outer_bbox(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        comb_outer_x1 = min(b1_x1, b2_x1)
        comb_outer_y1 = min(b1_y1, b2_y1)
        comb_outer_x2 = max(b1_x2, b2_x2)
        comb_outer_y2 = max(b1_y2, b2_y2)

        return comb_outer_x1, comb_outer_y1, comb_outer_x2, comb_outer_y2

    def get_bbox_area(self, box):
        x1, y1, x2, y2 = box
        b1_w = x2 - x1
        b1_h = y2 - y1

        # check for invalid boxes with negative width or height
        if b1_w <= 0.0 or b1_h <= 0.0:
            return 0.0

        return b1_w * b1_h

    def get_bboxes_IOU(self, box1, box2):
        # area of first box ...
        b1_area = self.get_bbox_area(box1)

        # area of second box ...
        b2_area = self.get_bbox_area(box2)

        # intersection between two boxes ...
        comb_inter = self.get_inter_bbox(box1, box2)
        inter_area = self.get_bbox_area(comb_inter)

        combined_union_area = b1_area + b2_area - inter_area

        # print((b1_area, b2_area, box1, box2, comb_inter, inter_area, combined_union_area))

        return inter_area / combined_union_area

    def get_combined_box_area_ratio(self, box1, box2):
        # area of first box ...
        b1_area = self.get_bbox_area(box1)

        # area of second box ...
        b2_area = self.get_bbox_area(box2)

        # intersection between two boxes ...
        comb_inter = self.get_inter_bbox(box1, box2)
        inter_area = self.get_bbox_area(comb_inter)

        combined_union_area = b1_area + b2_area - inter_area

        comb_outer = self.get_outer_bbox(box1, box2)
        outer_area = self.get_bbox_area(comb_outer)

        area_ratio = combined_union_area / outer_area

        return area_ratio

    def visualize_boxes(self, bboxes, color=(255,0, 0)):
        out_img = np.zeros((self.height, self.width, 3), np.uint8)

        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

        return out_img

    def spatial_box_grouping(self, frame_bboxes, min_combined_ratio=None):
        if min_combined_ratio is None:
            min_combined_ratio = self.min_combined_ratio

        current_boxes = list(frame_bboxes)
        merged_boxes = True

        while merged_boxes:
            merged_boxes = False

            # sort bounding boxes by descending size ...
            boxes_by_size = []
            for x1, y1, x2, y2 in current_boxes:
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                area = w * h

                boxes_by_size.append((area, (x1, y1, x2, y2)))

            boxes_by_size = sorted(boxes_by_size, reverse=True, key=lambda x:x[0])

            # print(boxes_by_size)

            # create interval index to find matches quicker (all boxes against all boxes from same frame)
            int_index_x = IntervalIndex(True)
            int_index_y = IntervalIndex(True)
            for box_idx, (area, (x1, y1, x2, y2)) in enumerate(boxes_by_size):
                int_index_x.add(x1, x2 + 1, box_idx)
                int_index_y.add(y1, y2 + 1, box_idx)

            # ... find pair-wise matches ...
            set_x = set(int_index_x.find_matches(int_index_x))
            set_y = set(int_index_y.find_matches(int_index_y))
            # .... list of all pairs of boxes with intersecting intervals in X and Y
            merge_candidates = sorted(list(set_x.intersection(set_y)))
            # ... filter self-matches and repetitions ...
            merge_candidates = [(box_idx1, box_idx2) for box_idx1, box_idx2 in merge_candidates if box_idx1 < box_idx2]
            # ... split by first box ..
            candidates_by_box = {idx: [] for idx in range(len(boxes_by_size))}
            for box_idx1, box_idx2 in merge_candidates:
                candidates_by_box[box_idx1].append(box_idx2)

            # print(merge_candidates)
            # print(candidates_by_box)

            box_added = [False] * len(boxes_by_size)
            current_boxes = []
            # for each box (sorted by size)
            for box_idx in range(len(boxes_by_size)):
                # if this box has been previously added (merged with ealier box)
                if box_added[box_idx]:
                    # skip ...
                    continue

                box_added[box_idx] = True

                # current box boundaries
                c_area, (c_x1, c_y1, c_x2, c_y2) = boxes_by_size[box_idx]

                for second_box_idx in candidates_by_box[box_idx]:
                    if box_added[second_box_idx]:
                        # skip merge candidate ...
                        continue

                    # get box boundaries ...
                    o_area, (o_x1, o_y1, o_x2, o_y2) = boxes_by_size[second_box_idx]

                    comb_area_ratio = self.get_combined_box_area_ratio((c_x1, c_y1, c_x2, c_y2),
                                                                       (o_x1, o_y1, o_x2, o_y2))

                    # print(((c_x1, c_y1, c_x2, c_y2), boxes_by_size[second_box_idx], comb_area_ratio, box_idx, second_box_idx))

                    if comb_area_ratio >= min_combined_ratio:
                        # merge!
                        # expand current bounding box to include the smaller box ...
                        c_x1 = min(c_x1, o_x1)
                        c_y1 = min(c_y1, o_y1)
                        c_x2 = max(c_x2, o_x2)
                        c_y2 = max(c_y2, o_y2)

                        # mark second box as added, so it won't be added to the current list ..
                        box_added[second_box_idx] = True

                        merged_boxes = True

                # add to the next set of accepted boxes
                current_boxes.append((c_x1, c_y1, c_x2, c_y2))

        """
        if len(frame_bboxes) > 30:
            original_img = self.visualize_boxes(frame_bboxes, (255, 0, 0))
            final_img = self.visualize_boxes(current_boxes, (0, 255, 0))
            final_img[:, :, 0] = original_img[:, :, 0]
            debug_img = cv2.resize(final_img,(960, 540))
            cv2.imshow("check", debug_img)
            cv2.waitKey()
            raise Exception("Error!")
        """

        return current_boxes

    def add_frame(self, frame_bboxes):
        current_bboxes = self.spatial_box_grouping(frame_bboxes, self.min_combined_ratio)

        current_bboxes_idxs = []
        if self.img_idx == 0:
            # simply copy all
            for bbox_id, bbox in enumerate(current_bboxes):
                # add the box to list of unique boxes ...
                self.unique_bbox_objects.append(bbox)

                # frames on which the bbox appears, raw label assigned to that CC
                self.unique_bbox_frames.append([(0, bbox_id)])

                bbox_idx = len(self.unique_bbox_objects) - 1
                current_bboxes_idxs.append((bbox_idx, bbox))

                self.bbox_last_frame.append(0)
                self.bbox_active.append(bbox_idx)

                # add to indices ...
                x1, y1, x2, y2 = bbox
                self.bbox_int_index_x.add(x1, x2, bbox_idx)
                self.bbox_int_index_y.add(y1, y2, bbox_idx)
        else:
            # create indices for current bboxes
            other_index_x = IntervalIndex(True)
            other_index_y = IntervalIndex(True)

            for bbox_idx, (x1, y1, x2, y2) in enumerate(current_bboxes):
                other_index_x.add(x1, x2, bbox_idx)
                other_index_y.add(y1, y2, bbox_idx)

            # compute CC with matching regions
            set_x = set(other_index_x.find_matches(self.bbox_int_index_x))
            set_y = set(other_index_y.find_matches(self.bbox_int_index_y))

            # list of all pairs of CC with intersecting intervals in X and Y
            merged = sorted(list(set_x.intersection(set_y)))
            self.tempo_count += len(merged)

            # check every matching CC
            pre_add_size = len(self.bbox_active)
            next_match_idx = 0

            for bbox_idx, bbox in enumerate(current_bboxes):
                found = False

                # check all matches in the list of matches for current CC
                while next_match_idx < len(merged) and merged[next_match_idx][0] == bbox_idx:
                    if not found:
                        prev_idx = merged[next_match_idx][1]
                        prev_bbox = self.unique_bbox_objects[prev_idx]

                        bbox_IOU = self.get_bboxes_IOU(bbox, prev_bbox)
                        # print(bbox_IOU)
                        if bbox_IOU >= self.min_temporal_IOU:
                            # assume they are equivalent
                            found = True
                            self.unique_bbox_frames[prev_idx].append((self.img_idx, bbox_idx))
                            current_bboxes_idxs.append((prev_idx, bbox))

                            # update last frame seen for this cc...
                            self.bbox_last_frame[prev_idx] = self.img_idx

                    next_match_idx += 1

                # Not match was found?
                if not found:
                    # add
                    self.unique_bbox_objects.append(bbox)
                    self.unique_bbox_frames.append([(self.img_idx, bbox_idx)])

                    new_bbox_idx = len(self.unique_bbox_objects) - 1
                    current_bboxes_idxs.append((new_bbox_idx, bbox))

                    self.bbox_last_frame.append(self.img_idx)
                    self.bbox_active.append(new_bbox_idx)

                    # add to indices ...
                    x1, y1, x2, y2 = bbox
                    self.bbox_int_index_x.add(x1, x2, new_bbox_idx)
                    self.bbox_int_index_y.add(y1, y2, new_bbox_idx)

            # remove CC that are no longer active
            pre_remove_size = len(self.bbox_active)
            tempo_pos = 0
            while tempo_pos < len(self.bbox_active):
                bbox_idx = self.bbox_active[tempo_pos]

                if self.img_idx - self.bbox_last_frame[bbox_idx] >= self.max_gap:
                    # no longer active ..
                    # delete from active list
                    del self.bbox_active[tempo_pos]

                    # delete from interval indices
                    bbox = self.unique_bbox_objects[bbox_idx]
                    x1, y1, x2, y2 = bbox
                    self.bbox_int_index_x.remove(x1, x2, bbox_idx)
                    self.bbox_int_index_y.remove(y1, y2, bbox_idx)
                    #print self.cc_last_frame[cc_idx],
                else:
                    # still active
                    tempo_pos += 1

            """
            total_added = pre_remove_size - pre_add_size
            total_removed = pre_remove_size - len(self.bbox_active)
            msg = "{0:d} , (Added: {1:d}, Removed: {2:d})".format(len(self.bbox_active), total_added, total_removed)
            print(msg)
            """

        self.bbox_idx_per_frame.append(current_bboxes_idxs)

        self.img_idx += 1

        if self.verbose:
            msg = "[{0:d} ({1:d}, {2:d})]".format(self.img_idx, len(current_bboxes), len(self.unique_bbox_objects))
            print(msg, end="\r")

    def finish_processing(self):
        if self.verbose:
            print(".")

    def split_stable_bboxes_by_gaps(self, max_gap, stable_min_frames):
        splitted_count = 0
        n_original_objects = len(self.unique_bbox_objects)
        for bbox_idx in range(n_original_objects):
            current_frames = self.unique_bbox_frames[bbox_idx]
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
                self.unique_bbox_frames[bbox_idx] = valid_groups[0]

                # for each group (new CC)
                for group_offset in range(1, len(valid_groups)):
                    new_bbox_idx = len(self.unique_bbox_objects)

                    # add another reference to the original CC
                    self.unique_bbox_objects.append(self.unique_bbox_objects[bbox_idx])

                    # add CC frame reference ...
                    self.unique_bbox_frames.append(valid_groups[group_offset])

                    # for each frame where the orginal appeared ..
                    for frame_idx, local_bbox_idx in valid_groups[group_offset]:
                        # find the CC on the frame ...
                        for offset, (global_bbox_idx, local_bbox) in enumerate(self.bbox_idx_per_frame[frame_idx]):
                            if global_bbox_idx == bbox_idx:
                                # replace
                                self.bbox_idx_per_frame[frame_idx][offset] = (new_bbox_idx, local_bbox)
                                break

                splitted_count += 1

        return splitted_count

    def get_stable_bbox_idxs(self, min_stable_frames):
        stable_idxs = []
        for bbox_idx in range(len(self.unique_bbox_objects)):
            if len(self.unique_bbox_frames[bbox_idx]) >= min_stable_frames:
                stable_idxs.append(bbox_idx)

        return stable_idxs

    def compute_overlapping_stable_bboxes(self, stable_idxs, temporal_window):
        n_objects = len(self.unique_bbox_objects)
        n_stable = len(stable_idxs)

        all_overlapping_cc = [[] for x in range(n_objects)]
        time_overlapping_cc = [[] for x in range(n_objects)]
        total_intersections = 0

        for offset1 in range(n_stable):
            # get first stable
            bbox_idx_1 = stable_idxs[offset1]
            bbox_1 = self.unique_bbox_objects[bbox_idx_1]

            bbox_1_t_start = self.unique_bbox_frames[bbox_idx_1][0][0]
            bbox_1_t_end = self.unique_bbox_frames[bbox_idx_1][-1][0]
            bbox_1_area = self.get_bbox_area(bbox_1)

            print("Processing: " + str(offset1),end="\r")

            for offset2 in range(offset1 + 1, n_stable):
                bbox_idx_2 = stable_idxs[offset2]
                bbox_2 = self.unique_bbox_objects[bbox_idx_2]

                bbox_2_t_start = self.unique_bbox_frames[bbox_idx_2][0][0]
                bbox_2_t_end = self.unique_bbox_frames[bbox_idx_2][-1][0]

                # check intersection in space
                IOU = self.get_bboxes_IOU(bbox_1, bbox_2)
                if IOU > 0.0001:
                    bbox_2_area = self.get_bbox_area(bbox_2)

                    inter_box_area = self.get_bbox_area(self.get_inter_bbox(bbox_1, bbox_2))

                    all_overlapping_cc[bbox_idx_1].append((bbox_idx_2, inter_box_area, bbox_2_area, bbox_1_area))
                    all_overlapping_cc[bbox_idx_2].append((bbox_idx_1, inter_box_area, bbox_1_area, bbox_2_area))

                    # now, check intersection in time (considering time window)
                    if ((bbox_1_t_end + temporal_window >= bbox_2_t_start) and
                        (bbox_2_t_end >= bbox_1_t_start - temporal_window)):

                        ratio_1 = inter_box_area / bbox_1_area
                        ratio_2 = inter_box_area / bbox_2_area

                        # they have some intersection in pixels....
                        time_overlapping_cc[bbox_idx_1].append((bbox_idx_2, ratio_1, ratio_2))
                        time_overlapping_cc[bbox_idx_2].append((bbox_idx_1, ratio_2, ratio_1))

                        total_intersections += 1

        return time_overlapping_cc, total_intersections, all_overlapping_cc

    def compute_groups(self, stable_idxs, overlapping_bboxes):
        n_stable = len(stable_idxs)

        # Determine Groups of CC's that coexist in time and space and treat them as single units...
        bboxes_groups = []
        group_idx_per_bbox = {}
        for offset1 in range(n_stable):
            bbox_idx_1 = stable_idxs[offset1]

            if bbox_idx_1 in group_idx_per_bbox:
                # use the existing group
                group_idx = group_idx_per_bbox[bbox_idx_1]
            else:
                # create new group
                group_idx = len(bboxes_groups)
                bboxes_groups.append([bbox_idx_1])
                group_idx_per_bbox[bbox_idx_1] = group_idx

            # for every CC that occupies the same space ..
            for bbox_idx_2, _, _ in overlapping_bboxes[bbox_idx_1]:
                # if it is not in the current group, add it ...
                if bbox_idx_2 not in group_idx_per_bbox:
                    # add to current group
                    group_idx_per_bbox[bbox_idx_2] = group_idx
                    # add to the current group
                    bboxes_groups[group_idx].append(bbox_idx_2)
                else:
                    other_group_idx = group_idx_per_bbox[bbox_idx_2]
                    if other_group_idx != group_idx:
                        # different group? merge ..
                        #print("Merging groups")
                        # for each element in the other group
                        for other_idx_cc in bboxes_groups[other_group_idx]:
                            # link to the current group
                            group_idx_per_bbox[other_idx_cc] = group_idx
                            # add to the current group
                            bboxes_groups[group_idx].append(other_idx_cc)

                        # leave the other group empty
                        bboxes_groups[other_group_idx] = []

        # clean up empty groups
        final_bbox_groups = []
        final_group_idx_per_bbox = {}

        for group in bboxes_groups:
            if len(group) > 0:
                new_group_idx = len(final_bbox_groups)
                final_bbox_groups.append(group)

                for box_idx in group:
                    final_group_idx_per_bbox[box_idx] = new_group_idx

        return final_bbox_groups, final_group_idx_per_bbox

    def compute_groups_temporal_information(self, bbox_groups):
        # compute temporal information for each group
        n_frames = len(self.bbox_idx_per_frame)
        group_ages = {}
        groups_per_frame = [[] for frame_idx in range(n_frames)]

        for group_idx, group in enumerate(bbox_groups):
            if len(group) == 0:
                continue

            current_ages = []
            for cc_idx in group:
                g_first = self.unique_bbox_frames[cc_idx][0][0]
                g_last = self.unique_bbox_frames[cc_idx][-1][0]

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

    def compute_conflicting_groups(self, stable_idxs, all_overlapping_bboxes, n_groups, group_idx_per_bbox):
        n_stable = len(stable_idxs)

        # for each stable CC
        conflicts = {group_idx:{} for group_idx in range(n_groups)}
        for offset1 in range(n_stable):
            bbox_idx_1 = stable_idxs[offset1]

            # for every CC that occupies the same space (same group or not)..
            for bbox_idx_2, matched_pixels, size_bbox_2, size_bbox_1 in all_overlapping_bboxes[bbox_idx_1]:

                # consider each link only once
                if bbox_idx_1 < bbox_idx_2:
                    # UNION - Intersection
                    unmatched_pixels = size_bbox_1 + size_bbox_2 - matched_pixels * 2

                    # check if they are on different groups
                    group_idx1 = group_idx_per_bbox[bbox_idx_1]
                    group_idx2 = group_idx_per_bbox[bbox_idx_2]
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

    def find_container_bbox(self, bboxes):
        g_x1, g_y1, g_x2, g_y2 = bboxes[0]
        for o_x1, o_y1, o_x2, o_y2 in bboxes[1:]:
            g_x1 = min(g_x1, o_x1)
            g_y1 = min(g_y1, o_y1)
            g_x2 = max(g_x2, o_x2)
            g_y2 = max(g_y2, o_y2)

        return g_x1, g_y1, g_x2, g_y2

    def refine_bboxes(self, bboxes_groups, group_ages, temporal_refinement):
        groups_bboxes = {}
        refined_bboxes = {}
        refined_per_group = {}

        debug_group_ids = []

        for group_idx, group in enumerate(bboxes_groups):
            if len(group) == 0:
                continue

            # initial box ...
            bbox_idx = group[0]
            g_x1, g_y1, g_x2, g_y2 = self.unique_bbox_objects[bbox_idx]

            # expand to contain all images from all boxes for all frames in the current group
            for bbox_idx in group:

                # stable bbox level ..
                frame_idx, local_bbox_idx = self.unique_bbox_frames[bbox_idx][0]
                _, (u_x1, u_y1, u_x2, u_y2) = self.bbox_idx_per_frame[frame_idx][local_bbox_idx]

                # expand the boxes to get the bigger box that contains all the per-frame variations
                for frame_idx, local_bbox_idx in self.unique_bbox_frames[bbox_idx]:
                    global_bbox_idx, local_bbox = self.bbox_idx_per_frame[frame_idx][local_bbox_idx]

                    o_x1, o_y1, o_x2, o_y2 = local_bbox

                    # stable bbox level ...
                    u_x1 = min(u_x1, o_x1)
                    u_y1 = min(u_y1, o_y1)
                    u_x2 = max(u_x2, o_x2)
                    u_y2 = max(u_y2, o_y2)

                    # group level ...
                    g_x1 = min(g_x1, o_x1)
                    g_y1 = min(g_y1, o_y1)
                    g_x2 = max(g_x2, o_x2)
                    g_y2 = max(g_y2, o_y2)

                # print((group_idx, bbox_idx, (u_x1, u_x2, u_y1, u_y2), (g_x1, g_x2, g_y1, g_y2)))

                refined_bboxes[bbox_idx] = (u_x1, u_y1, u_x2, u_y2)

            groups_bboxes[group_idx] = (g_x1, g_y1, g_x2, g_y2)

            # compute the refined boxes for different temporal segments ..
            current_images = []
            current_ages = group_ages[group_idx]

            if group_idx in debug_group_ids:
                print((group_idx, groups_bboxes[group_idx], group_ages[group_idx]))

            # ... for each temporal segment ...
            for t_segment in range(len(current_ages) - 1):
                t_start = current_ages[t_segment]
                t_end = current_ages[t_segment + 1]

                if temporal_refinement == BBoxStabilityEstimator.RefineAllBoxesAllSegments:
                    # use larger group on all segments ...
                    current_images.append([groups_bboxes[group_idx]])
                else:
                    # ... for each box in the group ...
                    visible_bboxes = []
                    for bbox_idx in group:
                        # ... check
                        bbox_frames = [(f_idx, local_bbox_idx) for f_idx, local_bbox_idx in
                                       self.unique_bbox_frames[bbox_idx] if t_start <= f_idx <= t_end]
                        count_bbox_frames = len(bbox_frames)

                        # only if this box has visible frames in current time
                        if count_bbox_frames > 0:
                            visible_bboxes.append(refined_bboxes[bbox_idx])

                    # merge visible boxes that have any overlap
                    visible_bboxes = self.spatial_box_grouping(visible_bboxes, 0.0)

                    if temporal_refinement == BBoxStabilityEstimator.RefinePerBoxPerSegment:
                        # use all the merged boxes separately
                        current_images.append(visible_bboxes)
                    else:
                        # find container for all boxes ...
                        current_images.append([self.find_container_bbox(visible_bboxes)])

            refined_per_group[group_idx] = current_images

        return refined_bboxes, groups_bboxes, refined_per_group

    def refined_per_frame(self, bbox_groups, groups_per_frame, group_ages, refined_per_group,
                          save_prefix=None, stable_min_frames=3):

        group_next_segment = [0 for group_idx in range(len(bbox_groups))]
        all_reconstructed = []

        for img_idx, groups_in_frame in enumerate(groups_per_frame):
            reconstructed = []

            for group_idx in groups_in_frame:
                # find corresponding image ....
                current_ages = group_ages[group_idx]
                while current_ages[group_next_segment[group_idx] + 1] < img_idx:
                    # move to the next segment
                    group_next_segment[group_idx] += 1

                # use image of selected segment ...
                segment_bboxes = refined_per_group[group_idx][group_next_segment[group_idx]]

                # add to the image
                reconstructed += segment_bboxes

            if not save_prefix is None:
                # draw the reconstructed/refined boxes in white ...
                img_boxes = self.visualize_boxes(reconstructed, (255, 255, 255))

                # add the original stable/unstable boxes
                img_stable = np.zeros((self.height, self.width, 3), np.uint8)
                img_unstable = np.zeros((self.height, self.width, 3), np.uint8)
                for bbox_idx, local_bbox in self.bbox_idx_per_frame[img_idx]:
                    x1, y1, x2, y2 = local_bbox

                    if len(self.unique_bbox_frames[bbox_idx]) < stable_min_frames:
                        # add unstable
                        # reconstructed[y1:y2 + 1, x1:x2 + 1, 2] = 255
                        cv2.rectangle(img_unstable, (x1, y1), (x2, y2), (255,255, 255), thickness=2)
                    else:
                        # add stable
                        cv2.rectangle(img_stable, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

                # add stable in green channel ...
                img_boxes[:, :, 1] = np.bitwise_or(img_stable[:, :, 0], img_boxes[:, :, 1])
                # add unstable in red channel
                img_boxes[:, :, 2] = np.bitwise_or(img_unstable[:, :, 0], img_boxes[:, :, 2])

                cv2.imwrite(save_prefix + str(img_idx) + ".png", img_boxes)

            all_reconstructed.append(reconstructed)

        return all_reconstructed

    def compute_group_images(self, bboxes_groups, group_ages, binary_frames, sum_threshold, use_global):
        group_images = {}
        group_boundaries = {}

        debug_group_ids = []

        # TODO: pending, check if grouping by ages could work ... maybe from previous process, avoid creating
        #       very close ages ... group these maybe????

        for group_idx, group in enumerate(bboxes_groups):
            if len(group) == 0:
                continue

            # initial box ...
            bbox_idx = group[0]
            g_x1, g_y1, g_x2, g_y2 = self.unique_bbox_objects[bbox_idx]

            # expand to contain all images from all boxes for all frames in the current group
            for bbox_idx in group:
                for frame_idx, local_bbox_idx in self.unique_bbox_frames[bbox_idx]:
                    global_bbox_idx, local_bbox = self.bbox_idx_per_frame[frame_idx][local_bbox_idx]

                    o_x1, o_y1, o_x2, o_y2 = local_bbox

                    g_x1 = min(g_x1, o_x1)
                    g_y1 = min(g_y1, o_y1)
                    g_x2 = max(g_x2, o_x2)
                    g_y2 = max(g_y2, o_y2)

            group_boundaries[group_idx] = (g_x1, g_x2, g_y1, g_y2)

            # size ...
            g_width = g_x2 - g_x1 + 1
            g_height = g_y2 - g_y1 + 1

            # compute the actual images using temporal information ..
            current_images = []
            current_ages = group_ages[group_idx]

            g_complete_mask = np.zeros((g_height, g_width), dtype=np.int32)
            g_complete_sum = np.zeros((g_height, g_width), dtype=np.float64)

            if group_idx in debug_group_ids:
                print((group_idx, group_boundaries[group_idx], group_ages[group_idx]))

            for t_segment in range(len(current_ages) - 1):
                t_start = current_ages[t_segment]
                t_end = current_ages[t_segment + 1]

                g_mask = np.zeros((g_height, g_width), dtype=np.int32)
                g_sum = np.zeros((g_height, g_width), dtype=np.float64)

                # get the sum of CCs for all frames where they appear ...
                count_visible = 0
                for bbox_idx in group:
                    # x1, y1, x2, y2 = self.unique_bbox_objects[bbox_idx]

                    # cc_first = self.unique_cc_frames[cc_idx][0][0]
                    # cc_last = self.unique_cc_frames[cc_idx][-1][0]
                    bbox_frames = [(f_idx, local_bbox_idx) for f_idx, local_bbox_idx in self.unique_bbox_frames[bbox_idx]
                                   if t_start <= f_idx < t_end]
                    count_bbox_frames = len(bbox_frames)

                    # only if this box has visible frames in current time
                    if count_bbox_frames > 0:
                        count_visible += 1

                        # for each frame ...
                        for f_idx, local_bbox_idx in bbox_frames:
                            # find the box for this frame ...
                            _, (x1, y1, x2, y2) = self.bbox_idx_per_frame[f_idx][local_bbox_idx]

                            offset_x = x1 - g_x1
                            offset_y = y1 - g_y1

                            b_w = x2 - x1 + 1
                            b_h = y2 - y1 + 1

                            # bbox cut ....
                            # ... image per pixel ....
                            mask_cut = g_sum[offset_y:offset_y + b_h, offset_x:offset_x + b_w]
                            global_mask_cut = g_complete_sum[offset_y:offset_y + b_h, offset_x:offset_x + b_w]

                            # print((x1, y1, x2, y2, b_w, b_h, offset_x, offset_y, g_x1, g_y1))

                            bin_cut = (binary_frames[f_idx][y1:y2 + 1, x1:x2 + 1] // 255)
                            mask_cut += bin_cut

                            # global image ...
                            global_mask_cut += bin_cut

                            # ... counter per pixel ...
                            mask_cut = g_mask[offset_y:offset_y + b_h, offset_x:offset_x + b_w]
                            # all bounding-box pixels add as many times as frames where same box appears
                            mask_cut += 1

                            # repeat for global image ...
                            mask_cut = g_complete_mask[offset_y:offset_y + b_h, offset_x:offset_x + b_w]
                            mask_cut += 1

                # if local temporal segments are requested ...
                if not use_global:
                    visible_pixels = g_mask > 0
                    g_sum[visible_pixels] /= g_mask[visible_pixels]
                    g_sum[g_sum < sum_threshold] = 0.0
                    g_sum[g_sum >= sum_threshold] = 1.0
                    g_sum *= 255
                    segment_img = g_sum.astype(np.uint8)

                    if group_idx in debug_group_ids:
                        cv2.imshow("Segment IMG #{0:d} ({1:d}-{2:d})".format(group_idx, t_start, t_end), segment_img)
                        cv2.waitKey()
                    # raise Exception("STOP!")

                    current_images.append(segment_img)

                # TODO: test just adding all pixels from region ...
                g_sum = np.zeros((g_height, g_width), dtype=np.float64)
                for f_idx in range(t_start, t_end + 1):
                    pass

            # if global temporal segments are requested ...
            if use_global:
                visible_pixels = g_complete_mask > 0
                g_complete_sum[visible_pixels] /= g_complete_mask[visible_pixels]
                g_complete_sum[g_complete_sum < sum_threshold] = 0.0
                g_complete_sum[g_complete_sum >= sum_threshold] = 1.0
                g_complete_sum *= 255
                segment_img = g_complete_sum.astype(np.uint8)

                if group_idx in debug_group_ids or True:
                    magnified_segment = cv2.resize(segment_img, (segment_img.shape[1] * 3, segment_img.shape[0] * 3))
                    cv2.imshow("complete img: #{0:d}".format(group_idx), magnified_segment)
                    cv2.waitKey()

                # raise Exception("STOP!")
                current_images = [segment_img] * (len(current_ages) - 1)

            group_images[group_idx] = current_images

        return group_images, group_boundaries

    def frames_from_groups(self, cc_groups, group_boundaries, groups_per_frame, group_ages, group_images,
                           save_prefix=None, stable_min_frames=3, show_unstable=True):

        group_next_segment = [0 for group_idx in range(len(cc_groups))]
        clean_reconstructed = []

        for img_idx, groups_in_frame in enumerate(groups_per_frame):
            reconstructed = np.zeros((self.height, self.width, 3), dtype=np.int32)

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
                reconstructed[g_min_y:g_max_y + 1, g_min_x:g_max_x + 1, 2] += segment_img

            reconstructed[reconstructed > 255] = 255
            reconstructed = reconstructed.astype(np.uint8)

            if show_unstable:
                img_stable = np.zeros((self.height, self.width, 3), np.uint8)
                img_unstable = np.zeros((self.height, self.width, 3), np.uint8)
                for bbox_idx, local_bbox in self.bbox_idx_per_frame[img_idx]:
                    x1, y1, x2, y2 = local_bbox

                    if len(self.unique_bbox_frames[bbox_idx]) < stable_min_frames:
                        # add unstable
                        # reconstructed[y1:y2 + 1, x1:x2 + 1, 2] = 255
                        cv2.rectangle(img_unstable, (x1, y1), (x2, y2), (255,255, 255), thickness=2)
                    else:
                        # add stable
                        cv2.rectangle(img_stable, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

                # add stable in green channel ...
                reconstructed[:, :, 1] = np.bitwise_or(img_stable[:, :, 0], reconstructed[:, :, 1])
                # add unstable in red channel
                reconstructed[:, :, 2] = np.bitwise_or(img_unstable[:, :, 0], reconstructed[:, :, 2])

            #print(group_next_segment)
            if not save_prefix is None:
                cv2.imwrite(save_prefix + str(img_idx) + ".png", reconstructed)

            flag, raw_data = cv2.imencode(".png", reconstructed[:,:, 0])
            clean_reconstructed.append(raw_data)


        return clean_reconstructed

