
import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation


class PatchSampling:
    @staticmethod
    def get_padded_image(raw_image, n_pixels):
        assert len(raw_image.shape) == 2
        h, w = raw_image.shape

        padded_image = np.zeros((h + n_pixels * 2, w + n_pixels * 2), raw_image.dtype)
        padded_image[n_pixels:n_pixels + h, n_pixels:n_pixels + w] = raw_image.copy()

        # top-left corner
        padded_image[0:n_pixels, 0:n_pixels] = raw_image[n_pixels - 1::-1, n_pixels - 1::-1].copy()
        # top-right corner
        padded_image[0:n_pixels, -n_pixels:] = raw_image[n_pixels - 1::-1, -1:-n_pixels - 1:-1].copy()
        # bottom-left corner
        padded_image[-n_pixels:, 0:n_pixels] = raw_image[-1:-n_pixels - 1:-1, n_pixels - 1::-1].copy()
        # bottom-right corner
        padded_image[-n_pixels:, -n_pixels:] = raw_image[-1:-n_pixels - 1:-1, -1:-n_pixels - 1:-1].copy()

        # left
        padded_image[n_pixels:-n_pixels, 0:n_pixels] = raw_image[:, n_pixels - 1::-1].copy()
        # right
        padded_image[n_pixels:-n_pixels, -n_pixels:] = raw_image[:, -1:-n_pixels - 1:-1].copy()
        # top
        padded_image[0:n_pixels, n_pixels:-n_pixels] = raw_image[n_pixels - 1::-1].copy()
        # bottom
        padded_image[-n_pixels:, n_pixels:-n_pixels] = raw_image[-1:-n_pixels - 1:-1].copy()

        return padded_image

    @staticmethod
    def generate_pixel_type_visualization(fullbackground_map, fg_pixels, bg_close_neighbors, bg_whiteboard):
        h, w = fullbackground_map.shape

        mixture = np.zeros((h, w, 3), np.uint8)
        mixture[fullbackground_map > 0, 0] = 0
        mixture[fullbackground_map > 0, 1] = 255
        mixture[fullbackground_map > 0, 2] = 0

        mixture[fg_pixels > 0, 0] = 255
        mixture[fg_pixels > 0, 1] = 0
        mixture[fg_pixels > 0, 2] = 0

        mixture[bg_close_neighbors > 0, 0] = 0
        mixture[bg_close_neighbors > 0, 1] = 0
        mixture[bg_close_neighbors > 0, 2] = 255

        mixture[bg_whiteboard > 0, 0] = 255
        mixture[bg_whiteboard > 0, 1] = 255
        mixture[bg_whiteboard > 0, 2] = 255

        return mixture

    @staticmethod
    def SampleEdgeFixBg(all_keyframes, all_preprocessed, patch_size, patches_per_frame, fg_proportion, bg_close_prop,
                    bg_board_prop, bg_close_neighborhood):
        # compute patches ...
        patches_labels = []
        patches_images = []
        lecture_labels = None
        lecture_images = None
        half_patch = (patch_size - 1) / 2

        last_lecture = None
        for idx, kf in enumerate(all_keyframes):
            print("Extracting patches from keyframe #" + str(idx), end="\r")
            h, w, _ = kf.raw_image.shape

            if kf.lecture != last_lecture:
                last_lecture = kf.lecture

                lecture_labels = []
                lecture_images = []

                patches_labels.append(lecture_labels)
                patches_images.append(lecture_images)

            total_background = kf.binary_image[:, :, 0].sum() / 255
            total_foreground = h * w - total_background

            sample_vals = sorted(np.random.random(patches_per_frame).tolist())
            prob_image = np.zeros((h, w), dtype=np.float64)

            # per-pixel probabilites per group ....
            # get object mask
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(patch_size), int(patch_size)))
            fullbackground_map = cv2.dilate(kf.object_mask.astype(np.uint8) * 255, strel)

            # compute background pixels neighboring foreground
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (bg_close_neighborhood, bg_close_neighborhood))
            fg_pixels = 255 - kf.binary_image[:, :, 0]

            dilated_map = cv2.dilate(fg_pixels, strel)
            bg_close_neighbors = dilated_map - fg_pixels

            # subtract the expanded background from both the fg pixels and bg close neighbors
            fg_overlap_mask = np.logical_and(fg_pixels, fullbackground_map)
            fg_pixels[fg_overlap_mask] = 0

            close_overlap_mask = np.logical_and(bg_close_neighbors, fullbackground_map)
            bg_close_neighbors[close_overlap_mask] = 0

            bg_whiteboard = np.ones((h, w), np.uint8) * 255
            bg_whiteboard = bg_whiteboard - fullbackground_map
            bg_whiteboard = bg_whiteboard - fg_pixels
            bg_whiteboard = bg_whiteboard - bg_close_neighbors

            # mix = generate_pixel_type_visualization(fullbackground_map, fg_pixels, bg_close_neighborhood, bg_whiteboard)
            # cv2.imwrite("DELETE_mixture_kf_" + str(idx) + ".png", mix)

            total_foreground = fg_pixels.sum() / 255
            total_bg_close = bg_close_neighbors.sum() / 255
            total_bg_other = fullbackground_map.sum() / 255
            total_bg_board = bg_whiteboard.sum() / 255

            fg_pixel_prob = fg_proportion / total_foreground
            bg_close_pixel_prob = ((1.0 - fg_proportion) * bg_close_prop) / total_bg_close
            bg_board_pixel_prob = ((1.0 - fg_proportion) * (1.0 - bg_close_prop) * bg_board_prop) / total_bg_board
            bg_other_pixel_prob = ((1.0 - fg_proportion) * (1.0 - bg_close_prop) * (1.0 - bg_board_prop)) / total_bg_other

            prob_image[fg_pixels > 0] = fg_pixel_prob
            prob_image[bg_close_neighbors > 0] = bg_close_pixel_prob
            prob_image[bg_whiteboard > 0] = bg_board_pixel_prob
            prob_image[fullbackground_map > 0] = bg_other_pixel_prob


            # un-comment to visualize probabilities for key-frames
            # print("Foreground Background: " + str(fg_pixel_prob))
            # print("Close Background: " + str(bg_close_pixel_prob))
            # print("Board Background: " + str(bg_board_pixel_prob))
            # print("Other Background: " + str(bg_other_pixel_prob))

            # tempo_norm = prob_image.max()
            # pixel_prob_visual = ((prob_image / tempo_norm) * 255).astype(dtype=np.uint8)
            # cv2.imwrite("DELETE_pixel_probs_kf_" + str(idx) + ".png", pixel_prob_visual)

            # generate a padded version of the raw image ....
            padded_image = PatchSampling.get_padded_image(all_preprocessed[idx], half_patch)
            gt_padded_image = PatchSampling.get_padded_image(kf.binary_image[:, :, 0], half_patch)

            # now, get the patches as requested ...
            start_val = 0.0
            sample_idx = 0
            tempo_view = np.ones(padded_image.shape, np.uint8) * 128
            tempo_label_view = np.zeros(kf.raw_image.shape, np.uint8)
            for row in range(0, h):
                for col in range(0, w):
                    end_val = start_val + prob_image[row, col]
                    while sample_idx < len(sample_vals) and sample_vals[sample_idx] < end_val:
                        # get sample,
                        local_patch = padded_image[row:row + patch_size, col:col + patch_size].copy()

                        # invert label, use 0 for background, 1 for foreground ...
                        pixel_label = int((255 - kf.binary_image[row, col, 0]) / 255)

                        # hard label for final pixel class
                        local_label = pixel_label

                        # un-comment to generate visualization of sampled patches
                        # tempo_view[row:row + patch_size, col:col + patch_size] = local_patch.copy()
                        # if local_label == 1:
                        #     tempo_label_view[row, col, :] = (255, 255, 255)
                        # else:
                        #     tempo_label_view[row, col, :] = (64, 255, 52)
                        lecture_labels.append(local_label)
                        lecture_images.append(local_patch)

                        # move to next value to sample...
                        sample_idx += 1

                    start_val = end_val
                    if sample_idx >= len(sample_vals):
                        break

                if sample_idx >= len(sample_vals):
                    break

            # visualization of sampled patches ...
            # cv2.imwrite("DELETE_KF_sampled_patches_" + str(idx) + ".png", tempo_view)
            # cv2.imwrite("DELETE_KF_sampled_labels_" + str(idx) + ".png", tempo_label_view)

        return patches_images, patches_labels

    @staticmethod
    def SampleEdgeContBg(all_keyframes, all_preprocessed, patch_size, patches_per_frame, fg_proportion):
        # compute patches ...
        patches_labels = []
        patches_images = []
        half_patch = (patch_size - 1) / 2

        last_lecture = None
        for idx, kf in enumerate(all_keyframes):
            print("Extracting patches from keyframe #" + str(idx), end="\r")
            h, w, _ = kf.raw_image.shape

            if kf.lecture != last_lecture:
                last_lecture = kf.lecture

                lecture_labels = []
                lecture_images = []

                patches_labels.append(lecture_labels)
                patches_images.append(lecture_images)

            # generate a padded version of the raw image ....
            padded_image = PatchSampling.get_padded_image(all_preprocessed[idx], half_patch)
            gt_padded_image = PatchSampling.get_padded_image(kf.binary_image[:, :, 0], half_patch)

            # get object mask
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(patch_size), int(patch_size)))
            fullbackground_map = cv2.dilate(kf.object_mask.astype(np.uint8) * 255, strel)

            fg_pixels = 255 - kf.binary_image[:, :, 0]
            bg_pixels = kf.binary_image[:, :, 0].copy()

            # subtract the expanded general background from the whiteboard pixels
            full_bg_mask = fullbackground_map > 0
            fg_pixels[full_bg_mask] = 0
            bg_pixels[full_bg_mask] = 0

            sample_vals = sorted(np.random.random(patches_per_frame).tolist())
            prob_image = np.zeros((h, w), dtype=np.float64)

            # per-pixel probabilites per group ....
            # (uniform foreground)
            total_foreground = fg_pixels.sum() / 255
            fg_pixel_prob = fg_proportion / total_foreground
            prob_image[fg_pixels > 0] = fg_pixel_prob

            # (non-uniform background.... stronger by context intensity)

            # ... get the edge intensity image minus general background
            edge_padded = all_preprocessed[idx].copy()
            # (avoid adding general background pixels to the sums)
            edge_padded[full_bg_mask] = 0
            edge_padded = PatchSampling.get_padded_image(edge_padded, half_patch)

            # ... compute window-based intensities ....
            all_windows = np.lib.stride_tricks.as_strided(edge_padded,  shape=[h, w, patch_size, patch_size],
                                                          strides=padded_image.strides + padded_image.strides)

            all_windows = all_windows.reshape((h, w, patch_size * patch_size))
            window_sums = np.sum(all_windows, axis=2)
            # (make the sums of general background pixels 0)
            window_sums[fullbackground_map > 0] = 0
            # for probability smoothing for whiteboard bg pixels ...
            window_sums[fullbackground_map == 0] += 1
            # ... remove foreground ...
            window_sums[fg_pixels > 0] = 0

            # tempo_norm  = window_sums.max()
            # tempo_visual = (window_sums * 255) / tempo_norm
            # cv2.imwrite("bg_window_sums_kf_" + str(idx) + ".png", tempo_visual.astype(np.uint8))

            bg_norm = window_sums.sum()
            bg_probs = (window_sums.astype(np.float64) / bg_norm) * (1.0 - fg_proportion)
            bg_prob_mask = bg_probs > 0
            prob_image[bg_prob_mask] = bg_probs[bg_prob_mask]

            # tempo_norm = prob_image.max()
            # pixel_prob_visual = ((prob_image / tempo_norm) * 255).astype(dtype=np.uint8)
            #cv2.imwrite("DELETE_pixel_probs_kf_" + str(idx) + ".png", pixel_prob_visual)

            # now, get the patches as requested ...
            start_val = 0.0
            sample_idx = 0
            tempo_view = np.ones(padded_image.shape, np.uint8) * 128
            tempo_label_view = np.zeros(kf.raw_image.shape, np.uint8)
            for row in range(0, h):
                for col in range(0, w):
                    end_val = start_val + prob_image[row, col]
                    while sample_idx < len(sample_vals) and sample_vals[sample_idx] < end_val:
                        # get sample,
                        local_patch = padded_image[row:row + patch_size, col:col + patch_size].copy()

                        # invert label, use 0 for background, 1 for foreground ...
                        pixel_label = int((255 - kf.binary_image[row, col, 0]) / 255)

                        # hard label for final pixel class
                        local_label = pixel_label

                        # un-comment to generate visualization of sampled patches
                        tempo_view[row:row + patch_size, col:col + patch_size] = local_patch.copy()
                        if local_label == 1:
                            tempo_label_view[row, col, :] = (255, 255, 255)
                        else:
                            tempo_label_view[row, col, :] = (64, 255, 52)

                        lecture_labels.append(local_label)
                        lecture_images.append(local_patch)

                        # move to next value to sample...
                        sample_idx += 1

                    start_val = end_val
                    if sample_idx >= len(sample_vals):
                        break

                if sample_idx >= len(sample_vals):
                    break

            # visualization of sampled patches ...
            # cv2.imwrite("DELETE_KF_sampled_patches_" + str(idx) + ".png", tempo_view)
            # cv2.imwrite("DELETE_KF_sampled_labels_" + str(idx) + ".png", tempo_label_view)

        return patches_images, patches_labels

    @staticmethod
    def SampleBinaryContBg(all_keyframes, patch_size, patches_per_frame, fg_proportion):
        # compute patches ...
        patches_labels = []
        patches_images = []
        half_patch = int((patch_size - 1) / 2)

        last_lecture = None
        for idx, kf in enumerate(all_keyframes):
            print("Extracting patches from keyframe #" + str(idx), end="\r")
            h, w, _ = kf.raw_image.shape

            if kf.lecture != last_lecture:
                last_lecture = kf.lecture

                lecture_labels = []
                lecture_images = []

                patches_labels.append(lecture_labels)
                patches_images.append(lecture_images)

            # get object mask
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(patch_size), int(patch_size)))
            fullbackground_map = cv2.dilate(kf.object_mask.astype(np.uint8) * 255, strel)

            fg_pixels = 255 - kf.binary_image[:, :, 0]
            bg_pixels = kf.binary_image[:, :, 0].copy()

            # subtract the expanded general background from the whiteboard pixels
            full_bg_mask = fullbackground_map > 0
            fg_pixels[full_bg_mask] = 0
            bg_pixels[full_bg_mask] = 0

            sample_vals = sorted(np.random.random(patches_per_frame).tolist())
            prob_image = np.zeros((h, w), dtype=np.float64)

            # per-pixel probabilites per group ....
            # (uniform foreground)
            total_foreground = fg_pixels.sum() / 255
            fg_pixel_prob = fg_proportion / total_foreground
            prob_image[fg_pixels > 0] = fg_pixel_prob

            # (non-uniform background.... stronger by context intensity)
            # ... get the binary intensity image minus general background

            bin_padded = fg_pixels.astype(np.float32).copy()
            # (avoid adding general background pixels to the sums)
            bin_padded[full_bg_mask] = 0

            if patch_size % 2 == 1:
                # half patch on ech side
                bin_padded = PatchSampling.get_padded_image(bin_padded, half_patch)
            else:
                # pass
                bin_padded = PatchSampling.get_padded_image(bin_padded, half_patch + 1)
                # remove 1px from left and top ...
                bin_padded = bin_padded[1:, 1:].copy()

            # ... compute window-based intensities ....
            all_windows = np.lib.stride_tricks.as_strided(bin_padded,  shape=[h, w, patch_size, patch_size],
                                                          strides=bin_padded.strides + bin_padded.strides)

            all_windows = all_windows.reshape((h, w, patch_size * patch_size))
            window_sums = np.sum(all_windows, axis=2)
            # (make the sums of general background pixels 0)
            window_sums[fullbackground_map > 0] = 0
            # for probability smoothing for whiteboard bg pixels ...
            window_sums[fullbackground_map == 0] += bin_padded.min()
            # ... remove foreground ...
            window_sums[fg_pixels > 0] = 0

            # tempo_norm = window_sums.max()
            # tempo_visual = (window_sums * 255) / tempo_norm
            # cv2.imwrite("DELETE_bg_window_sums_kf_" + str(idx) + ".png", tempo_visual.astype(np.uint8))

            bg_norm = window_sums.sum()
            bg_probs = (window_sums.astype(np.float64) / bg_norm) * (1.0 - fg_proportion)
            bg_prob_mask = bg_probs > 0

            prob_image[bg_prob_mask] = bg_probs[bg_prob_mask]

            # tempo_norm = prob_image.max()
            # pixel_prob_visual = ((prob_image / tempo_norm) * 255).astype(dtype=np.uint8)
            # cv2.imwrite("DELETE_pixel_probs_kf_" + str(idx) + ".png", pixel_prob_visual)

            # now, get the patches as requested ...
            start_val = 0.0
            sample_idx = 0
            bin_padded = bin_padded.astype(np.uint8)
            tempo_view = np.ones(bin_padded.shape, np.uint8) * 128
            tempo_label_view = np.zeros(kf.raw_image.shape, np.uint8)
            for row in range(0, h):
                for col in range(0, w):
                    end_val = start_val + prob_image[row, col]
                    while sample_idx < len(sample_vals) and sample_vals[sample_idx] < end_val:
                        # get sample,
                        local_patch = bin_padded[row:row + patch_size, col:col + patch_size].copy()

                        # invert label, use 0 for background, 1 for foreground ...
                        pixel_label = int((255 - kf.binary_image[row, col, 0]) / 255)

                        # hard label for final pixel class
                        local_label = pixel_label

                        # un-comment to generate visualization of sampled patches
                        tempo_view[row:row + patch_size, col:col + patch_size] = local_patch.copy()
                        if local_label == 1:
                            tempo_label_view[row, col, :] = (255, 255, 255)
                        else:
                            tempo_label_view[row, col, :] = (64, 255, 52)

                        lecture_labels.append(local_label)
                        lecture_images.append(local_patch)

                        # move to next value to sample...
                        sample_idx += 1

                    start_val = end_val
                    if sample_idx >= len(sample_vals):
                        break

                if sample_idx >= len(sample_vals):
                    break

            # visualization of sampled patches ...
            # cv2.imwrite("DELETE_KF_sampled_patches_" + str(idx) + ".png", tempo_view)
            # cv2.imwrite("DELETE_KF_sampled_labels_" + str(idx) + ".png", tempo_label_view)

        return patches_images, patches_labels

    @staticmethod
    def SampleBinaryCC(all_keyframes, patch_size, patches_per_frame, min_scale, max_scale):
        # compute patches ...
        patches_images = []
        last_lecture = None
        for idx, kf in enumerate(all_keyframes):
            assert isinstance(kf, KeyFrameAnnotation)

            print("Extracting patches from keyframe #" + str(idx), end="\r")

            if kf.lecture != last_lecture:
                last_lecture = kf.lecture
                lecture_images = []
                patches_images.append(lecture_images)

            valid_ccs_images = []
            for cc in kf.binary_cc:
                assert isinstance(cc, ConnectedComponent)

                longest = max(cc.getWidth(), cc.getHeight())
                scale = patch_size / longest
                if min_scale <= scale <= max_scale:
                    # normalize
                    cc.normalizeImage(patch_size)
                    # copy ...
                    valid_image = cc.normalized
                    # remove (to save memory)
                    cc.normalized = None

                    valid_image = valid_image.astype(np.uint8)
                    valid_ccs_images.append((cc, valid_image))

            if len(valid_ccs_images) == 0:
                continue

            # now, chose CC at random ...
            # some might be repeated with uniform probability ...
            sample_vals = (np.random.random(patches_per_frame) * len(valid_ccs_images)).astype(np.int32)
            sample_vals = sorted(sample_vals.tolist())

            tempo_view = np.ones(kf.binary_image.shape, np.uint8) * 128

            for sample_idx in sample_vals:
                sample_cc, sample_image = valid_ccs_images[sample_idx]
                lecture_images.append(sample_image)

                # add to visualization ...
                tempo_view_cut = tempo_view[sample_cc.min_y:sample_cc.max_y + 1, sample_cc.min_x:sample_cc.max_x + 1, :]
                tempo_view_cut[sample_cc.img > 0, :] = 255

            # visualization of sampled patches ...
            # cv2.imwrite("DELETE_KF_sampled_patches_" + str(idx) + ".png", tempo_view)

        return patches_images