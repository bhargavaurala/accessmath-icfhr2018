
import numpy as np

class Visualizer:

    @staticmethod
    def combine_bin_images_w_disp(binary1, binary2, delta_x, delta_y, content_lum=0):
        assert len(binary1.shape) == 2
        assert len(binary2.shape) == 2

        h, w = binary1.shape

        combined = np.zeros((h, w, 3), dtype=np.uint8)

        # use copies ...
        binary1 = binary1.copy()
        binary2_src = binary2.copy()

        # used to store displaced copy of second image
        binary2_dst = np.zeros((h, w), dtype=np.uint8)

        dest_min_x = max(0, delta_x)
        dest_max_x = w + min(0, delta_x)
        dest_min_y = max(0, delta_y)
        dest_max_y = h + min(0, delta_y)

        src_min_x = max(0, -delta_x)
        src_max_x = w + min(0, -delta_x)
        src_min_y = max(0, -delta_y)
        src_max_y = h + min(0, -delta_y)

        binary2_dst[dest_min_y:dest_max_y, dest_min_x:dest_max_x] = binary2_src[src_min_y:src_max_y, src_min_x:src_max_x]

        # identical pixels will be original colors (black/white)
        same_mask = binary2_dst == binary1
        combined[same_mask, 0] = binary1[same_mask]
        combined[same_mask, 1] = binary1[same_mask]
        combined[same_mask, 2] = binary1[same_mask]

        # pixels of just previous will be dark green
        diff_mask = np.logical_not(same_mask)
        only_2 = np.logical_and(diff_mask, binary2_dst == content_lum)
        combined[only_2, 0] = 0
        combined[only_2, 1] = 128
        combined[only_2, 2] = 0

        # pixels of just current will be dark red
        only_1 = np.logical_and(diff_mask, binary1 == content_lum)
        combined[only_1, 0] = 128
        combined[only_1, 1] = 0
        combined[only_1, 2] = 0

        return combined

    @staticmethod
    def show_keyframes_matches(height, width, exact, partial, unmatched_recall, unmatched_precision, disp_x, disp_y):
        """
        :param height: Image height
        :param width: Image width
        :param exact: List of exact matches between Ground Truth and summary (CCMatchInfo objects)
        :param partial: List of partial matches between Ground Truth and summary (CCMatchInfo objects)
        :param unmatched_recall: List of un-matched Ground Truth elements (ConnectedComponent objects)
        :param unmatched_precision: List of un-matched summary elements (ConnectedComponent objects)
        :param disp_x: SUMM X offset
        :param disp_y: SUMM y offset
        :return: Image showing matched and un-matched elements between Ground Truth key-frame and Summary key-frame
        """

        match_image = np.ones((height, width, 3), dtype=np.uint8) * 16

        # frame 1 -> usually ground truth
        # frame 2 -> usually summary frame
        mask = np.zeros((height, width), dtype=np.uint8)

        for cc in unmatched_recall:
            cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_cut[cc.img > 0] += 1

        for cc in unmatched_precision:
            cc_cut = mask[cc.min_y + disp_y:cc.max_y + disp_y + 1, cc.min_x + disp_x:cc.max_x + 1 + disp_x]
            cc_cut[cc.img[:cc_cut.shape[0], :cc_cut.shape[1]] > 0] += 2

        for e_match in exact:
            for cc in e_match.frame1_ccs_refs:
                cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                cc_cut[cc.img > 0] += 6

            for cc in e_match.frame2_ccs_refs:
                cc_cut = mask[cc.min_y + disp_y:cc.max_y + 1 + disp_y, cc.min_x + disp_x:cc.max_x + 1 + disp_x]
                cc_cut[cc.img[:cc_cut.shape[0], :cc_cut.shape[1]] > 0] += 7

        for p_match in partial:
            for cc in p_match.frame1_ccs_refs:
                cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                # cc_cut[cc.img > 0, :] = (0, 128, 255)
                cc_cut[cc.img > 0] += 4

            for cc in p_match.frame2_ccs_refs:
                cc_cut = mask[cc.min_y + disp_y:cc.max_y + 1 + disp_y, cc.min_x + disp_x:cc.max_x + 1 + disp_x]
                # cc_cut[cc.img[:cc_cut.shape[0], :cc_cut.shape[1]] > 0, :] = (32, 64, 128)
                cc_cut[cc.img[:cc_cut.shape[0], :cc_cut.shape[1]] > 0] += 5

        match_image[mask == 0, :] = (255, 255, 255)     # Empty background

        match_image[mask == 1, :] = (0, 0, 255)         # Failed GT
        match_image[mask == 2, :] = (15, 15, 15)        # Failed Summ
        match_image[mask == 3, :] = (204, 92, 63)       # Failed GT/Summ overlap.

        match_image[mask == 4, :] = (0, 0, 255)         # Partial GT
        match_image[mask == 5, :] = (15, 15, 15)        # Partial Summ
        match_image[mask == 9, :] = (204, 92, 63)       # Partial GT/Summ overlap.

        match_image[mask == 6, :] = (0, 0, 255)         # Exact GT
        match_image[mask == 7, :] = (15, 15, 15)        # Exact Summ
        match_image[mask == 13, :] = (15, 205, 0)       # Exact GT/Summ overlap.

        return match_image

    @staticmethod
    def show_gt_matches(height, width, exact, partial, unmatched):
        """
        :param height: Image height
        :param width: Image width
        :param exact: List of exact Ground Truth matches (ConnectedComponent objects)
        :param partial: List of partial Ground Truth matches (ConnectedComponent objects)
        :param unmatched: List of un-matched Ground Truth elements (ConnectedComponent objects)
        :return: Image showing globally matched and un-matched elements from Ground Truth key-frame
        """
        match_image = np.ones((height, width, 3), dtype=np.uint8) * 16

        # frame 1 -> usually ground truth
        # frame 2 -> usually summary frame
        mask = np.zeros((height, width), dtype=np.uint8)

        for cc in unmatched:
            cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_cut[cc.img > 0] += 1

        for cc in exact:
            cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_cut[cc.img > 0] += 2

        for cc in partial:
            cc_cut = mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_cut[cc.img > 0] += 3

        match_image[mask == 0, :] = (255, 255, 255)     # Empty background
        match_image[mask == 1, :] = (63, 92, 204)       # Failed GT
        match_image[mask == 2, :] = (76, 177, 34)       # Exact GT
        match_image[mask == 3, :] = (14, 201, 255)      # Partial GT

        return match_image

