
import cv2
import numpy as np
from AccessMath.preprocessing.data.visual_alignment import VisualAlignment

class FrameGTBinarizer:
    def __init__(self, gt_keyframes, gt_segments):
        self.width = 0
        self.height = 0

        self.frame_count = 0

        self.gt_keyframes = gt_keyframes
        self.gt_segments = gt_segments
        self.last_kf_idx = None

        # some general parameters
        self.frame_times = None
        self.frame_indices = None
        self.compressed_frames = None

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frame_count = 0
        self.last_kf_idx = 0

        self.frame_times = []
        self.frame_indices = []
        self.compressed_frames = []

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    # ==========================================================
    # based on:
    #  http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    # ==========================================================
    def findHomography(self, src_frame, dst_frame, ratio=0.7, min_matches=10):
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(src_frame, None)
        kp2, des2 = sift.detectAndCompute(dst_frame, None)

        index_params = {"algorithm" : 0, "trees": 5}
        search_params = {"checks": 50}

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test.
        good = [m for m, n in matches if m.distance < ratio * n.distance]

        if len(good) > min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            return H
        else:
            return None



    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        self.frame_count += 1

        # identify first the corresponding key-frames
        current_kf_idx = self.last_kf_idx
        while current_kf_idx < len(self.gt_keyframes) and abs_frame_idx >= self.gt_keyframes[current_kf_idx].idx:
            current_kf_idx += 1

        if current_kf_idx == 0:
            prev_kf = None
        else:
            prev_kf = self.gt_keyframes[current_kf_idx - 1]
        if current_kf_idx >= len(self.gt_keyframes):
            next_kf = None
        else:
            next_kf = self.gt_keyframes[current_kf_idx]

        self.last_kf_idx = current_kf_idx

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe.apply( image )
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if next_kf is not None:
            # to project from current frame space to key-frame space
            M = self.findHomography(gray_frame, next_kf.grayscale_image[:, :, 0])

            if M is not None:
                size = next_kf.raw_image.shape[1], next_kf.raw_image.shape[0]
                projected = cv2.warpPerspective(gray_frame, M, size)
                diff = np.abs(next_kf.grayscale_image[:, :, 0].astype(np.float32) - projected.astype(np.float32))

                print((diff.min(), diff.max(), np.median(diff), diff.mean()))

                # diff_mask = diff < 15 # 5 unstable - 25 kinda high - 50 too much
                diff_mask = diff < diff.mean() + diff.std() * 0.25

                base_binary = next_kf.binary_image[:, :, 0].copy()
                base_binary[np.logical_not(diff_mask)] = 255

                base_binary = 255 - base_binary

                final_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                for cc in next_kf.binary_cc:
                    base_cut = base_binary[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                    match = np.count_nonzero(np.logical_and(base_cut, cc.img))
                    if match / cc.size > 0.95:
                        cc_cut = final_binary[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                        cc_cut[cc.img > 0] = 255

                #cv2.imwrite("delete/frame_" + str(abs_frame_idx) + ".png", diff_mask.astype(np.uint8) * 255)
                cv2.imwrite("delete/frame_" + str(abs_frame_idx) + ".png", final_binary)


                #projected_bin = cv2.warpPerspective(next_kf.binary_image, M, size)

                #diff = np.abs(gray_frame.astype(np.float32) - projected_gray.astype(np.float32))

                #avg_r = diff[:, :, 0].mean()
                #avg_g = diff[:, :, 1].mean()
                #avg_b = diff[:, :, 2].mean()
                #print((avg_r, avg_g, avg_b))
                #diff = np.sqrt(np.sum(np.power(diff, 2), axis=2))
                #print((diff.min(), diff.max(), np.median(diff), diff.mean()))

                #tempo_frame = frame.copy().astype(np.float32)
                #tempo_frame[:, :, 0] -= avg_r
                #tempo_frame[:, :, 1] -= avg_g

                #tempo_frame[:, :, 2] -= avg_b
                #diff = tempo_frame.astype(np.float32) - projected.astype(np.float32)
                #avg_r = diff[:, :, 0].mean()
                #avg_g = diff[:, :, 1].mean()
                #avg_b = diff[:, :, 2].mean()
                #print((avg_r, avg_g, avg_b))

                #diff = np.sqrt(np.sum(np.power(diff, 2), axis=2))

                # filter_size = 11
                # kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
                # diff = cv2.filter2D(diff, -1, kernel)

                """
                tempo_vis = np.zeros(next_kf.raw_image.shape, dtype=np.uint8)
                tempo_vis[:, :, 0] = gray_frame
                # tempo_vis[:, :, 1] = next_kf.grayscale_image[:, :, 0]
                tempo_vis[:, :, 2] = cv2.cvtColor(projected, cv2.COLOR_RGB2GRAY)
                """


                #tempo_projected = projected.copy()
                #tempo_projected[np.logical_not(diff_mask), :] = 0

                # cv2.imwrite("delete/frame_" + str(abs_frame_idx) + ".png", projected_bin)

                # tempo_frame = frame.copy()
                # tempo_frame[np.logical_not(diff_mask), :] = 0

                #cv2.imshow("frame", frame)
                #cv2.imshow("GT", next_kf.raw_image)
                #cv2.imshow("Scaled diff", (diff / diff.max()))

                #cv2.imshow("GT mask", tempo_projected)
                #cv2.imshow("Frame mask", tempo_frame)
                #cv2.imshow("Binary", projected_bin)

                #cv2.waitKey()

                #x = 5 / 0

        pass


        """
        flag, raw_data = cv2.imencode(".png", binary)

        self.compressed_frames.append(raw_data)
        self.frame_indices.append(abs_frame_idx)
        self.frame_times.append(abs_time)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(binary)
        """

    def debug_frame(self, binary):
        out_name = self.debug_out_dir + "/binary_" + self.debug_video_name + "_" + str(self.frame_count) + ".png"
        cv2.imwrite(out_name, binary)

    def getWorkName(self):
        return "Frame Binarizer using Ground Truth"

    def finalize(self):
        pass