
from AM_CommonTools.data.connected_component import ConnectedComponent

class SpaceTimeStruct:
    def __init__(self, frame_times, frame_indices, frame_height, frame_width,
                 group_ages, group_images, group_boundaries):
        # information about the sample from the video ...
        self.frame_times = frame_times
        self.frame_indices = frame_indices

        self.width = frame_width
        self.height = frame_height

        self.cc_group_ages = group_ages
        self.cc_group_images = group_images
        self.cc_group_boundaries = group_boundaries

    def groups_in_frame_range(self, frame_start, frame_end, group_list=None):
        # apply as a filter of input list ...

        if group_list is None:
            # no input list specified, start with all groups ...
            group_list = list(self.cc_group_ages.keys())

        # copy elements that pass the filter ...
        result_list = []
        for group_idx in group_list:
            start_rel_idx = self.cc_group_ages[group_idx][0]
            end_rel_idx = self.cc_group_ages[group_idx][-1]

            if self.frame_indices[start_rel_idx] <= frame_end and frame_start <= self.frame_indices[end_rel_idx]:
                result_list.append(group_idx)

        return result_list

    def groups_in_space_region(self, r_min_x, r_max_x, r_min_y, r_max_y, group_list=None):
        # apply as a filter of input list ...

        if group_list is None:
            # no input list specified, start with all groups ...
            group_list = list(self.cc_group_ages.keys())

        # copy elements that pass the filter ...
        result_list = []
        for group_idx in group_list:
            g_min_x, g_max_x, g_min_y, g_max_y = self.cc_group_boundaries[group_idx]

            if (g_min_x <= r_max_x and r_min_x <= g_max_x) and (g_min_y <= r_max_y and r_min_y <= g_max_y):
                result_list.append(group_idx)

        return result_list

    def get_CC_instances(self, group_list, frame_idx):
        instances = []
        for group_idx in group_list:
            group_ages = self.cc_group_ages[group_idx]
            first_frame = self.frame_indices[group_ages[0]]
            last_frame = self.frame_indices[group_ages[-1]]
            if first_frame <= frame_idx <= last_frame:
                # find corresponding image
                interval_idx = 0
                while self.frame_indices[group_ages[interval_idx + 1]] < frame_idx:
                    interval_idx += 1
            elif frame_idx < first_frame:
                # use the first image ...
                interval_idx = 0
            else:
                # use the last image ..
                interval_idx = len(self.cc_group_images[group_idx]) - 1

            cc_img = self.cc_group_images[group_idx][interval_idx]
            min_x, max_x, min_y, max_y = self.cc_group_boundaries[group_idx]
            size = cc_img.sum() // 255
            instances.append(ConnectedComponent(0, min_x, max_x, min_y, max_y, size, cc_img))

        return instances

    def find_oldest_in_group(self, group_list):
        ages = sorted([(self.cc_group_ages[group_idx][0], group_idx) for group_idx in group_list])
        rel_idx, group_idx = ages[0]

        return group_idx, self.frame_indices[rel_idx], self.frame_times[rel_idx]



