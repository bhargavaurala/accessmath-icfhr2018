
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AccessMath.annotation.unique_cc_group import UniqueCCGroup
from AccessMath.preprocessing.content.aligner import Aligner

from AccessMath.util.visualizer import Visualizer

from .cc_match_info import CCMatchInfo
from .eval_parameters import EvalParameters


class Evaluator:
    @staticmethod
    def check_equivalent_cc(cc1, cc2, global_align, window, min_recall, min_precision):
        all_scores = []
        # evaluate matching score within a small window ...
        for loc_disp_y in range(-window, window + 1):
            for loc_disp_x in range(-window, window + 1):
                disp_y = global_align[3] + loc_disp_y
                disp_x = global_align[4] + loc_disp_x

                # apply displacement
                cc1.translateBox(disp_x, disp_y)

                # if boxes overlap after local displacement
                if ((cc1.min_x < cc2.max_x and cc2.min_x < cc1.max_x) and
                    (cc1.min_y < cc2.max_y and cc2.min_y < cc1.max_y)):
                    recall, precision = cc1.getOverlapFMeasure(cc2, False, False)

                    if recall + precision > 0.0:
                        fscore = (2.0 * recall * precision) / (recall + precision)
                    else:
                        fscore = 0.0

                    all_scores.append((fscore, recall, precision, loc_disp_x, loc_disp_y))

                # remove displacement
                cc1.translateBox(-disp_x, -disp_y)

        # boxes did not match ...
        if len(all_scores) == 0:
            return False

        # sort alignment scores. ...
        all_scores = sorted(all_scores, reverse=True, key=lambda x:x[0])
        # pick the highest ...
        fscore, recall, precision, loc_disp_x, loc_disp_y = all_scores[0]

        # check ...
        return recall >= min_recall and precision >= min_precision


    @staticmethod
    def keyframes_unique_cc(keyframe_set, alignments, local_window, min_recall, min_precision, verbose=False):
        percentiles = [25, 50, 75, 100]

        total_raw_cc = 0
        h, w, _ = keyframe_set[0].raw_image.shape
        all_sizes = []
        size_percentiles = []
        cc_groups = []
        for keyframe in keyframe_set:
            if keyframe.binary_cc is None:
                keyframe.update_binary_cc()

            total_raw_cc += len(keyframe.binary_cc)

            local_groups = {}
            for cc in keyframe.binary_cc:
                all_sizes.append(cc.size)
                local_groups[cc.strID()] = None

            cc_groups.append(local_groups)
            # print("   Key-frame: " + str(keyframe.idx) + ", found: " + str(len(current_cc)) + " CCs")

        if verbose:
            print("\tRaw CC count: " + str(total_raw_cc))
            # compute size statistics
            all_sizes = np.array(all_sizes)
            print("\t-> Percentiles:")
            for percent in percentiles:
                tempo_percentile = np.percentile(all_sizes, percent)
                size_percentiles.append(tempo_percentile)
                print("\t\t" + str(percent) + "%\t<\t" + str(tempo_percentile))

        # create mapping for first frame (everything is unique)
        unique_ccs = []
        active_ccs = []
        for cc in keyframe_set[0].binary_cc:
            new_group = UniqueCCGroup(cc, 0)

            unique_ccs.append(new_group)
            cc_groups[0][cc.strID()] = new_group
            active_ccs.append(new_group)

        for kf_idx in range(1, len(keyframe_set)):
            keyframe = keyframe_set[kf_idx]
            not_yet_found = list(active_ccs)
            active_ccs = []

            align = alignments[kf_idx - 1]
            # print((kf_idx, align))

            # for each cc in the current key-frame ....
            for kf_cc in keyframe.binary_cc:
                # check if similar enough to elements in active cc
                found = False
                for nyf_idx, active_cc in enumerate(not_yet_found):
                    if Evaluator.check_equivalent_cc(kf_cc, active_cc.cc_refs[-1], align, local_window, min_recall, min_precision):
                        # mark this CC as still active ...
                        active_ccs.append(active_cc)
                        # add cc instance to the group
                        active_cc.cc_refs.append(kf_cc)
                        # create inverse link to group
                        cc_groups[kf_idx][kf_cc.strID()] = active_cc
                        # remove from the list of CCs that still need to be found
                        del not_yet_found[nyf_idx]
                        # mark as found ...
                        found = True
                        break

                if not found:
                    new_group = UniqueCCGroup(kf_cc, kf_idx)
                    # add the new CC to the list of unique
                    unique_ccs.append(new_group)
                    # add inverse reference
                    cc_groups[kf_idx][kf_cc.strID()] = new_group
                    # mark as active ...
                    active_ccs.append(new_group)

        if verbose:
            print("\tUnique CC count: " + str(len(unique_ccs)))
            unique_sizes = np.array([group.cc_refs[0].size for group in unique_ccs])
            print("\t-> Percentiles:")
            for percent in percentiles:
                tempo_percentile = np.percentile(unique_sizes, percent)
                # size_percentiles.append(tempo_percentile)
                print("\t\t" + str(percent) + "%\t<\t" + str(tempo_percentile))

        return unique_ccs, cc_groups

    @staticmethod
    def keyframes_alignments(keyframes, window, min_fscore):
        alignments = []
        for idx in range(len(keyframes) - 1):
            curr_bin = keyframes[idx].binary_image[:, :, 0]
            next_bin = keyframes[idx + 1].binary_image[:, :, 0]

            align_info = Aligner.computeTranslationAlignment(curr_bin, next_bin, window, 0)
            align_fscore = align_info[0]

            if align_fscore < min_fscore:
                # reject alignment, asume content has changed
                # too much to be aligned with previous keyframe
                align_info = (0, 0, 0, 0, 0)

            alignments.append(align_info)

        return alignments

    @staticmethod
    def keyframes_overlapping_ccs(frame1_ccs, frame2_ccs, alignment, verbose=False):
        _, _, _, disp_y, disp_x = alignment

        overlapping_ccs = []

        # first, find the individual pairs of overlapping CCs
        for f2_cc in frame2_ccs:
            # apply displacement
            f2_cc.translateBox(disp_x, disp_y)

            for f1_cc in frame1_ccs:
                cc_recall, cc_precision = f1_cc.getOverlapFMeasure(f2_cc, False, False)

                if cc_recall > 0.0:
                    # overlap detected!
                    overlapping_ccs.append((f1_cc, f2_cc))

            # return CC to its original location ...
            f2_cc.translateBox(-disp_x, -disp_y)

        # then group them ....
        # first, assume all CCs are part of a group of their own
        overlap_frame1 = {cc.strID(): CCMatchInfo(cc, None) for cc in frame1_ccs}
        overlap_frame2 = {cc.strID(): CCMatchInfo(None, cc) for cc in frame2_ccs}

        # now, use pairs to merge them ....
        for f1_cc, f2_cc in overlapping_ccs:
            f1_cc_id = f1_cc.strID()
            f2_cc_id = f2_cc.strID()

            if overlap_frame1[f1_cc_id] != overlap_frame2[f2_cc_id]:
                # different, create a merged group, and redirect ...
                merged_overlap = CCMatchInfo.Merge(overlap_frame1[f1_cc_id], overlap_frame2[f2_cc_id])

                # link frame 1 elements to the new merged group
                for merged_f1_cc in merged_overlap.frame1_ccs_refs:
                    overlap_frame1[merged_f1_cc.strID()] = merged_overlap

                # link frame 2 elements to the new merged group
                for merged_f2_cc in merged_overlap.frame2_ccs_refs:
                    overlap_frame2[merged_f2_cc.strID()] = merged_overlap

        # get the final set of overlapping ccs (or candidate matches)
        overlap_set = list(set.union(set(overlap_frame1.values()), set(overlap_frame2.values())))

        if verbose:
            print("\t-> Count of raw pair-wise overlaps: " + str(len(overlapping_ccs)))
            print("\t-> total overlapping groups: " + str(len(overlap_set)))

        # return groups ...
        return overlap_set

    @staticmethod
    def match_overlapping_ccs(overlap_set, alignment, min_recall, min_precision, verbose=False):
        _, _, _, disp_y, disp_x = alignment

        exact_matches = []
        partial_matches = []
        unmatched_frame1 = []
        unmatched_frame2 = []

        for match_info in overlap_set:
            # check what type of candidate match it is
            if len(match_info.frame1_ccs_refs) == 0:
                # un-matched element from frame 2
                unmatched_frame2 += match_info.frame2_ccs_refs
            elif len(match_info.frame2_ccs_refs) == 0:
                # un-matched element from frame 1
                unmatched_frame1 += match_info.frame1_ccs_refs
            elif len(match_info.frame1_ccs_refs) == 1 and len(match_info.frame2_ccs_refs) == 1:
                # candidates for exact match ...
                f1_cc = match_info.frame1_ccs_refs[0]
                f2_cc = match_info.frame2_ccs_refs[0]

                # compute matching
                f2_cc.translateBox(disp_x, disp_y)
                cc_recall, cc_precision = f1_cc.getOverlapFMeasure(f2_cc, False, False)
                f2_cc.translateBox(-disp_x, -disp_y)

                # classify according to matching result ...
                if cc_recall >= min_recall and cc_precision >= min_precision:
                    exact_matches.append(match_info)
                else:
                    # add un-matched CCs ...
                    unmatched_frame1 += match_info.frame1_ccs_refs
                    unmatched_frame2 += match_info.frame2_ccs_refs
            else:
                # candidates for partial match ...
                combined_frame1 = ConnectedComponent.Merge(match_info.frame1_ccs_refs)
                combined_frame2 = ConnectedComponent.Merge(match_info.frame2_ccs_refs)

                # compute matching
                combined_frame2.translateBox(disp_x, disp_y)
                cc_recall, cc_precision = combined_frame1.getOverlapFMeasure(combined_frame2, False, False)

                # overlap_image = combined_frame1.getOverlapImage(combined_frame2)
                # print((cc_recall, cc_precision))
                # cv2.imshow("one_match", overlap_image)
                # cv2.waitKey()

                # classify according to matching result ...
                if cc_recall >= min_recall and cc_precision >= min_precision:
                    partial_matches.append(match_info)
                else:
                    # add un-matched CCs ...
                    unmatched_frame1 += match_info.frame1_ccs_refs
                    unmatched_frame2 += match_info.frame2_ccs_refs

        if verbose:
            print("\t-> Total exact matches: " + str(len(exact_matches)))
            print("\t-> Total partial matches groups: " + str(len(partial_matches)))
            print("\t-> Total CC in 1 unmatched: " + str(len(unmatched_frame1)))
            print("\t-> Total CC in 2 unmatched: " + str(len(unmatched_frame2)))

        return exact_matches, partial_matches, unmatched_frame1, unmatched_frame2

    @staticmethod
    def find_ccs_overlapping_background(gt_keyframe, summ_keyframe, alignment, verbose):
        _, _, _, disp_y, disp_x = alignment

        overlapping_ccs = []
        for f2_cc in summ_keyframe.binary_cc:
            # get id (before translation that changes id)
            cc_id = f2_cc.strID()
            # translate
            f2_cc.translateBox(disp_x, disp_y)
            if gt_keyframe.check_cc_overlaps_background(f2_cc):
                overlapping_ccs.append(cc_id)
            # remove translation
            f2_cc.translateBox(-disp_x, -disp_y)

        return overlapping_ccs


    @staticmethod
    def parallel_keyframe_align(candidate_data):
        gt_segment_idx, summ_segment_idx, gt_bin, summ_bin, window = candidate_data

        # find best recall-based alignment between key-frames
        align_info = Aligner.computeTranslationAlignment(gt_bin, summ_bin, window, 0, 1)

        # temp_image = Visualizer.combine_bin_images_w_disp(gt_bin, summ_bin, align_info[4], align_info[3], 0)
        # cv2.imwrite("tempo_disp_" + str(gt_segment_idx) + "_" + str(summ_segment_idx) +".png", temp_image)

        return gt_segment_idx, summ_segment_idx, align_info

    @staticmethod
    def summary_overlapping_ccs(gt_segments, gt_keyframes, summ_segments, summ_keyframes, window, min_align_recall,
                                verbose=False):
        # there has to be exactly one key-frame per segment
        # identify overlapping segments between summaries (that will be used to match key-frames)
        gt_segment_idx = 0 # 0
        summ_segment_idx = 0 # 0

        # <TEMPORAL>
        # gt_bin = gt_keyframes[2].binary_image[:, :, 0]
        # summ_bin = summ_keyframes[905].binary_image[:, :, 0]
        # align_info = Aligner.computeTranslationAlignment(gt_bin, summ_bin, window, 0, 1)
        # bg_overlaps = Evaluator.find_ccs_overlapping_background(gt_keyframes[2], summ_keyframes[905], align_info, verbose)
        # </TEMPORAL>

        all_overlapping_ccs = []
        background_overlaps = [{cc.strID(): 0 for cc in keyframe.binary_cc} for keyframe in summ_keyframes]

        if verbose:
            print("Finding alignment candidates")

        overlapping_data = []
        while gt_segment_idx < len(gt_segments) and summ_segment_idx < len(summ_segments):
            if (gt_segments[gt_segment_idx][0] < summ_segments[summ_segment_idx][1] and
                summ_segments[summ_segment_idx][0] < gt_segments[gt_segment_idx][1]):

                gt_bin = gt_keyframes[gt_segment_idx].binary_image[:, :, 0]
                summ_bin = summ_keyframes[summ_segment_idx].binary_image[:, :, 0]

                overlapping_data.append((gt_segment_idx, summ_segment_idx, gt_bin, summ_bin, window))

            # advance the segment with earliest termination...
            if summ_segments[summ_segment_idx][1] < gt_segments[gt_segment_idx][1]:
                summ_segment_idx += 1
            else:
                gt_segment_idx += 1

        if verbose:
            print("Starting alignment process")

        # compute alignments in parallel ...
        with ProcessPoolExecutor(max_workers=EvalParameters.UniqueCC_max_workers) as executor:
            for align_data in executor.map(Evaluator.parallel_keyframe_align, overlapping_data):
                gt_segment_idx, summ_segment_idx, align_info = align_data

                if verbose:
                    print("Computing overlaps GT KF #" + str(gt_segment_idx) + " - KF #" + str(summ_segment_idx))

                if align_info[1] < min_align_recall:
                    if verbose:
                        print("\t-> Recall is to low, skipping ...")
                else:
                    # find groups of overlapping CCs
                    gt_ccs = gt_keyframes[gt_segment_idx].binary_cc
                    summ_ccs = summ_keyframes[summ_segment_idx].binary_cc
                    overlapping_ccs = Evaluator.keyframes_overlapping_ccs(gt_ccs, summ_ccs, align_info, verbose)

                    # compute here summary CC's that fall in GT background CC
                    bg_overlaps = Evaluator.find_ccs_overlapping_background(gt_keyframes[gt_segment_idx],
                                                                            summ_keyframes[summ_segment_idx],
                                                                            align_info, verbose)
                    for cc_id in bg_overlaps:
                        background_overlaps[summ_segment_idx][cc_id] += 1

                    all_overlapping_ccs.append((gt_segment_idx, summ_segment_idx, align_info, overlapping_ccs))

        return all_overlapping_ccs, background_overlaps

    @staticmethod
    def find_gt_unique_cc_matches(gt_keyframes, gt_groups, gt_cc_group, summ_keyframes, all_overlapping_ccs,
                                  min_recall, min_precision, verbose=False):
        # for summary, count matches per key-frame
        summ_matches = [{cc.strID(): [] for cc in keyframe.binary_cc} for keyframe in summ_keyframes]

        # for ground truth, count matches per group (unique CC)
        gt_matches = {group.strID(): [] for group in gt_groups}
        frame_gt_matches = [{cc.strID(): [] for cc in keyframe.binary_cc} for keyframe in gt_keyframes]

        for gt_segment_idx, summ_segment_idx, align_info, overlapping_ccs in all_overlapping_ccs:
            if verbose:
                print("Computing matches GT KF #" + str(gt_segment_idx) + " - KF #" + str(summ_segment_idx))

            # now, evaluate matches
            match_res = Evaluator.match_overlapping_ccs(overlapping_ccs, align_info, min_recall, min_precision, verbose)
            exact, partial, failed_gt, failed_summ = match_res

            # add matches to the counts
            # ... exact ...
            for e_match in exact:
                # ground truth matches (the list should have just one element)
                for cc in e_match.frame1_ccs_refs:
                    gt_matches[gt_cc_group[gt_segment_idx][cc.strID()].strID()].append(e_match)
                    frame_gt_matches[gt_segment_idx][cc.strID()].append(e_match)

                # summary matches (the list should have just one element)
                for cc in e_match.frame2_ccs_refs:
                    summ_matches[summ_segment_idx][cc.strID()].append(e_match)

            # ... partial ...
            for p_match in partial:
                # ground truth matches
                for cc in p_match.frame1_ccs_refs:
                    gt_matches[gt_cc_group[gt_segment_idx][cc.strID()].strID()].append(p_match)
                    frame_gt_matches[gt_segment_idx][cc.strID()].append(p_match)

                # summary matches
                for cc in p_match.frame2_ccs_refs:
                    summ_matches[summ_segment_idx][cc.strID()].append(p_match)

            # un-comment to visualize matches ....
            """
            h, w, _ = gt_keyframes[gt_segment_idx].binary_image.shape

            img_matches = Visualizer.show_keyframes_matches(h, w, exact, partial, failed_gt, failed_summ,
                                                            align_info[4], align_info[3])
            img_name = "TEMPO_match_{0:d}_{1:d}_r_{2:.2f}_{3:.2f}.png".format(gt_segment_idx, summ_segment_idx,
                                                                              min_recall, min_precision)
            cv2.imwrite(img_name, img_matches)
            """


        return gt_matches, frame_gt_matches, summ_matches

    @staticmethod
    def match_list_type_counts(matches_lists):
        exact_matches, partial_matches, unmatched = 0, 0, 0
        for match_list in matches_lists:
            # check if was matched ....
            if len(match_list) == 0:
                unmatched += 1
            else:
                # partial or exact match ...
                exact_found = False
                for match in match_list:
                    if len(match.frame1_ccs_refs) == 1 and len(match.frame2_ccs_refs) == 1:
                        # one to one exact match ...
                        exact_found = True
                        break

                if exact_found:
                    exact_matches += 1
                else:
                    partial_matches += 1

        return exact_matches, partial_matches, unmatched

    @staticmethod
    def match_list_types(matches_per_cc):
        exact_matches, partial_matches, unmatched = [], [], []
        for cc_id in matches_per_cc:
            match_list = matches_per_cc[cc_id]

            # check if was matched ....
            if len(match_list) == 0:
                unmatched.append(cc_id)
            else:
                # partial or exact match ...
                exact_found = False
                for match in match_list:
                    if len(match.frame1_ccs_refs) == 1 and len(match.frame2_ccs_refs) == 1:
                        # one to one exact match ...
                        exact_found = True
                        break

                if exact_found:
                    exact_matches.append(cc_id)
                else:
                    partial_matches.append(cc_id)

        return exact_matches, partial_matches, unmatched

    @staticmethod
    def compute_unique_cc_summary_metrics(group_matches, per_frame_matches):
        # compute global metrics ....
        match_info = Evaluator.match_list_type_counts([group_matches[group_id] for group_id in group_matches])
        exact_matches, partial_matches, not_matched = match_info

        total = len(group_matches)
        if total > 0:
            only_exact_recall = exact_matches / total
            only_partial_recall = partial_matches / total
            recall = (exact_matches + partial_matches) / total
        else:
            only_exact_recall = 0.0
            only_partial_recall = 0.0
            recall = 0.0

        # compute per frame averages ...
        all_only_exact_recall, all_only_partial_recall, all_recall = [], [], []
        for kf_idx in range(len(per_frame_matches)):
            match_list = [per_frame_matches[kf_idx][cc_id] for cc_id in per_frame_matches[kf_idx]]
            match_info = Evaluator.match_list_type_counts(match_list)
            kf_exact_matches, kf_partial_matches, kf_not_matched = match_info

            kf_total = kf_exact_matches + kf_partial_matches + kf_not_matched

            if kf_total > 0:
                all_only_exact_recall.append(kf_exact_matches / kf_total)
                all_only_partial_recall.append(kf_partial_matches / kf_total)
                all_recall.append((kf_exact_matches + kf_partial_matches) / kf_total)
            else:
                all_only_exact_recall.append(1.0)
                all_only_partial_recall.append(0.0)
                all_recall.append(1.0)

        avg_only_exact_recall = np.array(all_only_exact_recall).mean()
        avg_only_partial_recall = np.array(all_only_partial_recall).mean()
        avg_recall = np.array(all_recall).mean()

        metrics = {
            "count": total,
            "recall": recall,
            "only_exact_recall": only_exact_recall,
            "only_partial_recall": only_partial_recall,

            "avg_only_exact_recall": avg_only_exact_recall,
            "avg_only_partial_recall": avg_only_partial_recall,
            "avg_recall": avg_recall,

            "partial_matches": partial_matches,
            "exact_matches": exact_matches,
            "unmatched": not_matched,
        }

        return metrics

    @staticmethod
    def compute_per_frame_summary_metrics(per_frame_matches, bg_overlaps):
        #  Key-frame stats (Precision?)
        total_count = 0
        exact_matches, partial_matches, not_matched, bg_not_matched = [], [], [], []
        all_precision, all_only_exact_precision, all_only_partial_precision = [], [], []
        prc_bg_not_matched, all_no_bg_precision = [], []

        for kf_idx in range(len(per_frame_matches)):
            match_list = [per_frame_matches[kf_idx][cc_id] for cc_id in per_frame_matches[kf_idx]]
            match_info = Evaluator.match_list_type_counts(match_list)
            kf_exact_matches, kf_partial_matches, kf_not_matched = match_info

            kf_bg_not_matched = 0
            for cc_id in per_frame_matches[kf_idx]:
                if len(per_frame_matches[kf_idx][cc_id]) == 0 and bg_overlaps[kf_idx][cc_id] > 0:
                    # not-matched and overlaps with background:
                    kf_bg_not_matched += 1

            exact_matches.append(kf_exact_matches)
            partial_matches.append(kf_partial_matches)
            not_matched.append(kf_not_matched)
            bg_not_matched.append(kf_bg_not_matched)

            kf_total = kf_exact_matches + kf_partial_matches + kf_not_matched
            total_count += kf_total
            if kf_total > 0:
                all_only_exact_precision.append(kf_exact_matches / kf_total)
                all_only_partial_precision.append(kf_partial_matches / kf_total)
                all_precision.append((kf_exact_matches + kf_partial_matches) / kf_total)
            else:
                all_only_exact_precision.append(1.0)
                all_only_partial_precision.append(0.0)
                all_precision.append(1.0)

            kf_no_bg_total = kf_total - kf_bg_not_matched
            if kf_no_bg_total > 0:
                all_no_bg_precision.append((kf_exact_matches + kf_partial_matches) / kf_no_bg_total)
            else:
                all_no_bg_precision.append(0.0)

            if kf_not_matched > 0:
                prc_bg_not_matched.append(kf_bg_not_matched / kf_not_matched)
            else:
                # nothing extra matching the background in this key-frame
                prc_bg_not_matched.append(0.0)

        avg_prc_bg_not_matched = np.array(prc_bg_not_matched).mean()
        avg_only_exact_precision = np.array(all_only_exact_precision).mean()
        avg_only_partial_precision = np.array(all_only_partial_precision).mean()
        avg_precision = np.array(all_precision).mean()
        avg_no_bg_precision = np.array(all_no_bg_precision).mean()

        total_exact_matches = sum(exact_matches)
        total_partial_matches = sum(partial_matches)
        total_not_matched = sum(not_matched)
        total_bg_not_matched = sum(bg_not_matched)

        if total_count > 0:
            only_exact_precision = total_exact_matches / total_count
            only_partial_precision = total_partial_matches / total_count
            precision = (total_exact_matches + total_partial_matches) / total_count
        else:
            only_exact_precision = 0.0
            only_partial_precision = 0.0
            precision = 0.0

        if total_count - total_bg_not_matched > 0:
            no_bg_precision = (total_exact_matches + total_partial_matches) / (total_count - total_bg_not_matched)
        else:
            no_bg_precision = 0.0


        if total_not_matched > 0:
            global_bg_not_matched = total_bg_not_matched / total_not_matched
        else:
            global_bg_not_matched = 0.0

        metrics = {
            "count": total_count,
            "avg_only_exact_precision": avg_only_exact_precision,
            "avg_only_partial_precision": avg_only_partial_precision,
            "avg_precision": avg_precision,
            "avg_prc_bg_not_matched": avg_prc_bg_not_matched,
            "avg_no_bg_precision": avg_no_bg_precision,

            "precision": precision,
            "only_exact_precision": only_exact_precision,
            "only_partial_precision": only_partial_precision,
            "global_bg_unmatched": global_bg_not_matched,
            "no_bg_precision": no_bg_precision,

            # these are lists ...
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "unmatched": not_matched,
            "bg_unmatched": bg_not_matched,

            "all_precision": all_precision,
            "all_only_exact_precision": all_only_exact_precision,
            "all_only_partial_precision": all_only_partial_precision,
            "all_no_bg_precision": all_no_bg_precision,
        }

        return metrics

    @staticmethod
    def filter_matches_per_size(gt_keyframes, gt_groups, gt_matches, frame_gt_matches, summ_keyframes, summ_matches,
                                bound_min, bound_max):
        filtered_gt_matches = {}
        groups_inv_index = {group.strID(): group for group in gt_groups}
        for group_id in gt_matches:
            if bound_min <= groups_inv_index[group_id].cc_refs[0].size < bound_max:
                filtered_gt_matches[group_id] = gt_matches[group_id]

        filtered_frame_gt_matches = []
        for kf_idx, keyframe in enumerate(gt_keyframes):
            filtered_frame_gt_matches.append({})
            for cc in keyframe.binary_cc:
                cc_id = cc.strID()
                if bound_min <= cc.size < bound_max:
                    filtered_frame_gt_matches[kf_idx][cc_id] = frame_gt_matches[kf_idx][cc_id]

        filtered_summ_matches = []
        for kf_idx, keyframe in enumerate(summ_keyframes):
            filtered_summ_matches.append({})
            for cc in keyframe.binary_cc:
                cc_id = cc.strID()
                if bound_min <= cc.size < bound_max:
                    filtered_summ_matches[kf_idx][cc_id] = summ_matches[kf_idx][cc_id]

        return filtered_gt_matches, filtered_frame_gt_matches, filtered_summ_matches

    @staticmethod
    def visualize_gt_matches(gt_keyframes, frame_gt_matches, img_prefix):
        # Visualizer.show_keyframes_matches(self)

        # for each gt key-frame
        for gt_kf_idx in range(len(frame_gt_matches)):
            exact_ids, partial_ids, unmatched_ids = Evaluator.match_list_types(frame_gt_matches[gt_kf_idx])

            ccs_by_id = gt_keyframes[gt_kf_idx].get_CCs_by_ID()

            exact = [ccs_by_id[cc_id] for cc_id in exact_ids]
            partial = [ccs_by_id[cc_id] for cc_id in partial_ids]
            unmatched = [ccs_by_id[cc_id] for cc_id in unmatched_ids]

            h, w, _ = gt_keyframes[gt_kf_idx].binary_image.shape

            img_matches = Visualizer.show_gt_matches(h, w, exact, partial, unmatched)

            img_name = "{0:s}_{1:d}.png".format(img_prefix, gt_kf_idx)
            cv2.imwrite(img_name, img_matches)


    @staticmethod
    def compute_summary_metrics(gt_segments, gt_keyframes, gt_groups, gt_cc_group, summ_segments, summ_keyframes,
                                verbose=False, gt_visual_prefix=None):

        global_window = EvalParameters.UniqueCC_global_tran_window
        min_align_r = EvalParameters.UniqueCC_min_align_recall

        size_percentiles = EvalParameters.UniqueCC_size_percentiles
        all_sizes = np.array([group.cc_refs[0].size for group in gt_groups])
        # res = plt.hist(sizes, 10, normed=0, facecolor='green', alpha=0.75)
        # plt.show()

        # define the boundaries for evaluation metrics per-size
        size_boundaries = [0]
        if EvalParameters.Report_Summary_Show_stats_per_size:
            for percentiles in size_percentiles:
                size_boundaries.append(int(round(np.percentile(all_sizes, percentiles))))
            size_boundaries.append(all_sizes.max() + 1)

        overlapping_ccs, bg_overlaps = Evaluator.summary_overlapping_ccs(gt_segments, gt_keyframes, summ_segments,
                                                                         summ_keyframes, global_window, min_align_r,
                                                                         verbose)

        metrics = {}
        sorted_range_names = []
        for min_r, min_p in zip(EvalParameters.UniqueCC_min_recall, EvalParameters.UniqueCC_min_precision):
            # get metrics for this combination of min Recall/Precision
            match_data = Evaluator.find_gt_unique_cc_matches(gt_keyframes, gt_groups, gt_cc_group, summ_keyframes,
                                                             overlapping_ccs, min_r, min_p, False)

            gt_matches, frame_gt_matches, summ_matches = match_data

            # if visualization must be generated ...
            if gt_visual_prefix is not None:
                # visualization output ...
                vis_dir = "{0:s}/{1:.2f}_{2:.2f}".format(gt_visual_prefix, min_r, min_p)
                # ... create directories (if they are not there already)
                os.makedirs(vis_dir, exist_ok=True)

                # ... generate and save images ....
                image_prefix = "{0:s}/match_".format(vis_dir)
                Evaluator.visualize_gt_matches(gt_keyframes, frame_gt_matches, image_prefix)

            for range_idx in range(len(size_boundaries)):

                if range_idx == len(size_boundaries) - 1:
                    current_range = "all"
                    range_gt_matches = gt_matches
                    range_frame_gt_matches = frame_gt_matches
                    range_summ_matches = summ_matches
                else:
                    current_range = "[{0}, {1})".format(size_boundaries[range_idx], size_boundaries[range_idx + 1])
                    range_matches = Evaluator.filter_matches_per_size(gt_keyframes, gt_groups, gt_matches,
                                                                      frame_gt_matches, summ_keyframes, summ_matches,
                                                                      size_boundaries[range_idx],
                                                                      size_boundaries[range_idx + 1])

                    range_gt_matches, range_frame_gt_matches, range_summ_matches = range_matches

                # Ground Truth stats (Recall)
                gt_metrics = Evaluator.compute_unique_cc_summary_metrics(range_gt_matches, range_frame_gt_matches)

                summ_metrics = Evaluator.compute_per_frame_summary_metrics(range_summ_matches, bg_overlaps)

                if not current_range in metrics:
                    sorted_range_names.append(current_range)
                    metrics[current_range] = []

                metrics[current_range].append({
                    "min_cc_recall": min_r,
                    "min_cc_precision": min_p,

                    "recall_metrics": gt_metrics,
                    "precision_metrics": summ_metrics,
                })

        return metrics, sorted_range_names

    @staticmethod
    def print_summary_recall_metrics(scope_metrics, scope):
        count_row = "{0:.2f}\t{1:.2f}\t|\t{2}\t|\t{3}\t{4}\t{5}\t{6}"
        percent_row = "{0:.2f}\t{1:.2f}\t|\t{2:.2f}\t|\t{3:.2f}\t{4:.2f}"
        if EvalParameters.Report_Summary_Show_Counts:
            print("Matching Params\t|\tGround Truth Matches (Count - " + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only\tMiss\tTotal")

            for all_metrics in scope_metrics:
                metrics = all_metrics["recall_metrics"]

                print(count_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                       metrics["exact_matches"] + metrics["partial_matches"], metrics["exact_matches"],
                                       metrics["partial_matches"], metrics["unmatched"], metrics["count"]))

        if EvalParameters.Report_Summary_Show_AVG_per_frame:
            print("")
            print("Matching Params\t|\tGround Truth Matches (Per Frame Recall - " + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only")
            for all_metrics in scope_metrics:
                metrics = all_metrics["recall_metrics"]

                print(percent_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                         metrics["avg_recall"] * 100.0, metrics["avg_only_exact_recall"] * 100.0,
                                         metrics["avg_only_partial_recall"] * 100.0))

        if EvalParameters.Report_Summary_Show_Globals:
            print("")
            print("Matching Params\t|\tGround Truth Matches (Unique CC Recall - " + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only")
            for all_metrics in scope_metrics:
                metrics = all_metrics["recall_metrics"]

                print(percent_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                         metrics["recall"] * 100.0, metrics["only_exact_recall"] * 100.0,
                                         metrics["only_partial_recall"] * 100.0))

    @staticmethod
    def print_summary_precision_metrics(scope_metrics, scope):
        count_row = "{0:.2f}\t{1:.2f}\t|\t{2}\t|\t{3}\t{4}\t{5}\t{6}\t{7}"
        percent_row = "{0:.2f}\t{1:.2f}\t|\t{2:.2f}\t|\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}"

        if EvalParameters.Report_Summary_Show_Counts:
            print("")
            print("Matching Params\t|\tSummary Matches (Count - " + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only\tMiss\tBG. Miss\tTotal")
            for all_metrics in scope_metrics:
                metrics = all_metrics["precision_metrics"]

                summ_total_exact = sum(metrics["exact_matches"])
                summ_total_partial = sum(metrics["partial_matches"])
                summ_total_unmatched = sum(metrics["unmatched"])
                summ_total_bg_unmatched = sum(metrics["bg_unmatched"])
                summ_total_cc = summ_total_exact + summ_total_partial + summ_total_unmatched
                print(count_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                       summ_total_exact + summ_total_partial, summ_total_exact, summ_total_partial,
                                       summ_total_unmatched, summ_total_bg_unmatched, summ_total_cc))

        if EvalParameters.Report_Summary_Show_AVG_per_frame:
            print("")
            print("Matching Params\t|\tSummary Matches (AVG Precision per Frame -" + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only\tBG. %\tNo BG P.")
            for all_metrics in scope_metrics:
                metrics = all_metrics["precision_metrics"]

                print(percent_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                         metrics["avg_precision"] * 100.0, metrics["avg_only_exact_precision"] * 100.0,
                                         metrics["avg_only_partial_precision"] * 100.0,
                                         metrics["avg_prc_bg_not_matched"] * 100.0,
                                         metrics["avg_no_bg_precision"] * 100.0))

        if EvalParameters.Report_Summary_Show_Globals:
            print("")
            print("Matching Params\t|\tSummary Matches (Global Precision -" + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only\tBG. %\tNo BG P.")
            for all_metrics in scope_metrics:
                metrics = all_metrics["precision_metrics"]

                print(percent_row.format(all_metrics["min_cc_recall"] * 100.0, all_metrics["min_cc_precision"] * 100.0,
                                         metrics["precision"] * 100.0, metrics["only_exact_precision"] * 100.0,
                                         metrics["only_partial_precision"] * 100.0,
                                         metrics["global_bg_unmatched"] * 100.0,
                                         metrics["no_bg_precision"] * 100.0))

    @staticmethod
    def print_compact_CC_metrics(scope_metrics, scope):

        headers = "Min_R\tMin_P"

        data_values = []
        for all_metrics in scope_metrics:
            metrics = all_metrics["recall_metrics"]

            data_values.append("{0:.2f}\t{1:.2f}".format(all_metrics["min_cc_recall"] * 100.0,
                                                            all_metrics["min_cc_precision"] * 100.0))

        # First, all Recall metrics
        count_row = "\t{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}"
        percent_row = "\t{0:.2f}\t{1:.2f}\t{2:.2f}"

        if EvalParameters.Report_Summary_Show_Counts:
            headers += "\tR_CT_EP\tR_CT_E\tR_CT_P\tR_CT_M\tR_CT_T"


            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["recall_metrics"]

                data_values[offset] += count_row.format(metrics["exact_matches"] + metrics["partial_matches"],
                                                        metrics["exact_matches"], metrics["partial_matches"],
                                                        metrics["unmatched"], metrics["count"])

        if EvalParameters.Report_Summary_Show_AVG_per_frame:
            headers += "\tR_AVG_EP\tR_AVG_E\tR_AVG_P"

            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["recall_metrics"]

                data_values[offset] += percent_row.format(metrics["avg_recall"] * 100.0,
                                                          metrics["avg_only_exact_recall"] * 100.0,
                                                          metrics["avg_only_partial_recall"] * 100.0)

        if EvalParameters.Report_Summary_Show_Globals:
            headers += "\tR_GBL_EP\tR_GBL_E\tR_GBL_P"

            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["recall_metrics"]

                data_values[offset] += percent_row.format(metrics["recall"] * 100.0,
                                                          metrics["only_exact_recall"] * 100.0,
                                                          metrics["only_partial_recall"] * 100.0)

        # Now, all Precision metrics
        count_row = "\t{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}\t{5:d}"
        percent_row = "\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}"

        if EvalParameters.Report_Summary_Show_Counts:
            headers += "\tP_CT_EP\tP_CT_E\tP_CT_P\tP_CT_M\tP_CT_BG_M\tP_CT_T"

            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["precision_metrics"]

                summ_total_exact = sum(metrics["exact_matches"])
                summ_total_partial = sum(metrics["partial_matches"])
                summ_total_unmatched = sum(metrics["unmatched"])
                summ_total_bg_unmatched = sum(metrics["bg_unmatched"])
                summ_total_cc = summ_total_exact + summ_total_partial + summ_total_unmatched

                data_values[offset] += count_row.format(summ_total_exact + summ_total_partial, summ_total_exact,
                                                        summ_total_partial, summ_total_unmatched,
                                                        summ_total_bg_unmatched, summ_total_cc)

        if EvalParameters.Report_Summary_Show_AVG_per_frame:
            headers += "\tP_AVG_EP\tP_AVG_E\tP_AVG_P\tP_AVG_BGP\tP_AVG_NBG"

            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["precision_metrics"]

                data_values[offset] += percent_row.format(metrics["avg_precision"] * 100.0,
                                                          metrics["avg_only_exact_precision"] * 100.0,
                                                          metrics["avg_only_partial_precision"] * 100.0,
                                                          metrics["avg_prc_bg_not_matched"] * 100.0,
                                                          metrics["avg_no_bg_precision"] * 100.0)

        if EvalParameters.Report_Summary_Show_Globals:
            print("")
            print("Matching Params\t|\tSummary Matches (Global Precision -" + scope + ")")
            print("Min. R.\tMin. P.\t|\tE + P\t|\tE. Only\tP. Only\tBG. %\tNo BG P.")

            headers += "\tP_GBL_EP\tP_GBL_E\tP_GBL_P\tP_GBL_BGP\tP_GBL_NBG"

            for offset, all_metrics in enumerate(scope_metrics):
                metrics = all_metrics["precision_metrics"]

                data_values[offset] += percent_row.format(metrics["precision"] * 100.0,
                                                          metrics["only_exact_precision"] * 100.0,
                                                          metrics["only_partial_precision"] * 100.0,
                                                          metrics["global_bg_unmatched"] * 100.0,
                                                          metrics["no_bg_precision"] * 100.0)

        # Print the final summary
        print(headers)

        for offset, all_metrics in enumerate(scope_metrics):
            print(data_values[offset])


    @staticmethod
    def compute_pixel_binary_metrics(gt_frames, summary_frames):

        all_recall, all_precision, all_fmeasure, all_board_precision, all_board_fmeasure = [], [], [], [], []
        for idx, gt_frame in enumerate(gt_frames):
            summ_frame = summary_frames[idx]

            gt_bin = 255 - gt_frame.binary_image[:, :, 0]
            summ_bin = 255 - summ_frame.binary_image[:, :, 0]

            # missing = gt_bin.copy()
            # missing[summ_bin > 0] = 0

            total_fg = gt_bin.sum() / 255
            total_summ_fg = summ_bin.sum() / 255
            total_correct = summ_bin[gt_bin > 0].sum() / 255

            only_board = summ_bin.copy()
            only_board[gt_frame.object_mask] = 0.0

            total_board_fg = only_board.sum()/ 255

            recall = total_correct / total_fg
            precision = total_correct / total_summ_fg
            board_precision = total_correct / total_board_fg
            if recall + precision > 0:
                fmeasure = (2.0 * recall * precision) / (recall + precision)
            else:
                fmeasure = 0.0

            if recall + board_precision > 0.0:
                board_fmeasure = (2.0 * recall * board_precision) / (recall + board_precision)
            else:
                board_fmeasure = 0.0

            all_recall.append(recall)
            all_precision.append(precision)
            all_fmeasure.append(fmeasure)
            all_board_precision.append(board_precision)
            all_board_fmeasure.append(board_fmeasure)

        return {
            "recall": np.array(all_recall).mean(),
            "precision": np.array(all_precision).mean(),
            "fmeasure": np.array(all_fmeasure).mean(),
            "board_precision": np.array(all_board_precision).mean(),
            "board_fmeasure": np.array(all_board_fmeasure).mean(),
        }
