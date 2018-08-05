
import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AccessMath.data.space_time_struct import SpaceTimeStruct
from AccessMath.preprocessing.content.cc_stability_estimator import CCStabilityEstimator


class KeyframeExtractor:

    @staticmethod
    def GenerateFromST3DForIntervals(st3D, video_segments, verbose=True):
        assert isinstance(st3D, SpaceTimeStruct)
        final_keyframes = []
        keyframes_times = []

        if verbose:
            print("Total CC Groups Given: " + str(len(st3D.cc_group_boundaries)))
            print("Total Video Segments: " + str(len(video_segments)))

        for segment_idx, (start_int, end_int) in enumerate(video_segments):
            if verbose:
                print("Processing segment #{0:d} ({1:d} - {2:d})".format(segment_idx + 1, start_int, end_int))

            local_times = []

            # locate all CC Groups that existed in this segment
            segment_group_ids = []
            segment_group_as_CC = {}
            for group_idx in st3D.cc_group_ages:
                if start_int <= st3D.cc_group_ages[group_idx][-1] and st3D.cc_group_ages[group_idx][0] <= end_int:
                    segment_group_ids.append(group_idx)

                    # choose the last image of the group that overlaps this interval
                    last_overlap = 0
                    while (last_overlap + 2 < len(st3D.cc_group_ages[group_idx]) and
                            st3D.cc_group_ages[group_idx][last_overlap + 2] <= end_int):
                        # next interval ...
                        last_overlap += 1

                    min_x, max_x, min_y, max_y = st3D.cc_group_boundaries[group_idx]

                    group_image = st3D.cc_group_images[group_idx][last_overlap]
                    group_size = group_image.sum() // 255

                    segment_group_as_CC[group_idx] = ConnectedComponent(group_idx, min_x, max_x, min_y, max_y,
                                                                        group_size, group_image)

            # detect conflicts ...
            group_CCs_list = list(segment_group_as_CC.values())
            overlapping_groups, no_overlaps = CCStabilityEstimator.compute_overlapping_CC_groups(group_CCs_list)

            # generate image ... first, add groups without conflicts ...
            frame_image = np.zeros((st3D.height, st3D.width, 3), dtype=np.uint8)
            frame_mask = np.zeros((st3D.height, st3D.width), dtype=np.int32)
            for offset in no_overlaps:
                group_cc = group_CCs_list[offset]

                frame_mask[group_cc.min_y:group_cc.max_y + 1, group_cc.min_x:group_cc.max_x + 1] += group_cc.img // 255
                # store time information associated with this CC ...
                start_time = st3D.frame_times[st3D.cc_group_ages[group_cc.cc_id][0]]
                # Time stamp + boundaries(X, Y) ...
                local_times.append((start_time, group_cc.min_x, group_cc.max_x, group_cc.min_y, group_cc.max_y))

            total_in_conflict = 0
            for conflict_idx, group in enumerate(overlapping_groups):
                total_in_conflict += len(group)

                if verbose:
                    print("... Conflict group # " + str(conflict_idx + 1))

                incompatible_matrix = np.zeros((len(group), len(group)), dtype=np.bool)
                sorted_by_age = []
                for overlap_idx,offset in enumerate(group):
                    group_cc = group_CCs_list[offset]

                    sorted_by_age.append((st3D.cc_group_ages[group_cc.cc_id][0], overlap_idx))

                    for sub_offset, offset2 in enumerate(group[overlap_idx + 1:]):
                        overlap_idx2 = sub_offset + overlap_idx + 1

                        group_cc2 = group_CCs_list[offset2]
                        recall, precision = group_cc.getOverlapFMeasure(group_cc2, False, False)

                        if recall > 0.0:
                            incompatible_matrix[overlap_idx, overlap_idx2] = True
                            incompatible_matrix[overlap_idx2, overlap_idx] = True

                    if verbose:
                        print("----> {0:d} - [{1:d}, {2:d}]".format(group_cc.cc_id,
                                                                    st3D.cc_group_ages[group_cc.cc_id][0],
                                                                    st3D.cc_group_ages[group_cc.cc_id][-1]))

                    # frame_mask[group_cc.min_y:group_cc.max_y + 1, group_cc.min_x:group_cc.max_x + 1] += (group_cc.img // 255) * 2
                    """
                    frame_cut = frame_image[group_cc.min_y:group_cc.max_y + 1, group_cc.min_x:group_cc.max_x + 1, :]
                    frame_cut[:, :, overlap_idx % 3] = group_cc.img.copy()
                    """

                sorted_by_age = sorted(sorted_by_age, reverse=True)

                accepted_recent = []
                for group_age, overlap_idx in sorted_by_age:
                    # check if incompatible with any of the accepted so far ...
                    compatible = True
                    for accepted_overlap_idx in accepted_recent:
                        if incompatible_matrix[accepted_overlap_idx, overlap_idx]:
                            compatible = False
                            break

                    if compatible:
                        accepted_recent.append(overlap_idx)

                accepted_groups_ccs = [group_CCs_list[group[overlap_idx]] for overlap_idx in accepted_recent]
                if verbose:
                    print("----> Will accept: " + ",".join([str(cc.cc_id) for cc in accepted_groups_ccs]))

                # finally add to the key-frame
                for group_cc in accepted_groups_ccs:
                    frame_mask[group_cc.min_y:group_cc.max_y + 1, group_cc.min_x:group_cc.max_x + 1] += group_cc.img // 255
                    # store time information associated with this CC ...
                    start_time = st3D.frame_times[st3D.cc_group_ages[group_cc.cc_id][0]]
                    # Time stamp + boundaries(X, Y) ...
                    local_times.append((start_time, group_cc.min_x, group_cc.max_x, group_cc.min_y, group_cc.max_y))

            frame_image[frame_mask == 1, :] = 255
            frame_image[frame_mask >= 2, 0] = 255 # Output Conflicts in different color

            #cv2.imwrite("TEMPO_" + str(tempo_rval) + "_" + str(segment_idx) + ".png", frame_image)

            frame_image[frame_mask >= 2, :] = 255

            if verbose:
                print("-> Total Groups contained: " + str(len(segment_group_ids)))
                print("-> Total Groups without Conflicts: " + str(len(no_overlaps)))
                print("-> Total Groups with Conflicts: " + str(total_in_conflict))

            final_keyframes.append(255 - frame_image)

            local_times = sorted(local_times)
            keyframes_times.append(local_times)

        return final_keyframes, keyframes_times

    @staticmethod
    def extract(binary_images, video_segments, treshold_length, verbose=False, save_prefix=None):
        # debug, show last image of each interval
        out_segments = []

        height, width = binary_images[0].shape

        for segment_idx, (start_int, end_int) in enumerate(video_segments):
            local_sum = np.zeros((height, width), dtype=np.float32)
            local_age = np.zeros((height, width), dtype=np.float32)
            local_last = np.zeros((height, width), dtype=np.float32)
            current_mask = np.zeros((height, width), dtype=np.bool)
            local_max_content = None
            local_max_count = None
            if verbose:
                print("Processing segment #" + str(segment_idx))

            for idx in range(start_int, end_int + 1):
                if local_max_content is None:
                    local_max_content = binary_images[idx]
                    local_max_count = np.count_nonzero(local_max_content)
                else:
                    new_max_count = np.count_nonzero(binary_images[idx])
                    if new_max_count > local_max_count:
                        local_max_content = binary_images[idx]
                        local_max_count = new_max_count

                image = binary_images[idx] / 255

                # check which pixels are new ...
                new_mask = image > 0

                # update the last time a pixel was modified ...
                local_last[new_mask] = idx

                # mark only pixels updated for the first time ...
                new_mask[current_mask] = 0

                # mark age of new pixels ...
                local_age[new_mask] = idx

                # update mask
                current_mask[new_mask] = True

                local_sum += image

            filtered_image = (local_sum >= treshold_length).astype(np.uint8) * 255

            current_segment = {
                "start": start_int,
                "end": end_int,
                "sum": local_sum,
                "age": local_age,
                "filtered": filtered_image,
                "local_max": local_max_content,
            }
            out_segments.append(current_segment)

            if save_prefix is not None:
                # output .... for debugging purposes
                sum_image = local_sum / local_sum.max()
                sum_image *= 255

                age_image = ((local_age - start_int) / (end_int - start_int)) * 255

                """
                density_img = np.zeros(filtered_image.shape, np.float32)
                mask = local_sum > 0
                density_img[mask] = (local_last[mask] -  local_age[mask]) / local_sum[mask]
                density_img *= 255.0
                """

                #cv2.imwrite(save_prefix +  "_sum_seg_" + str(segment_idx + 1) + ".png", sum_image)
                #cv2.imwrite(save_prefix +  "_age_seg_" + str(segment_idx + 1) + ".png", age_image)
                cv2.imwrite(save_prefix +  "_filt_seg" + "_" + str(segment_idx + 1) + ".png", filtered_image)

        return out_segments

