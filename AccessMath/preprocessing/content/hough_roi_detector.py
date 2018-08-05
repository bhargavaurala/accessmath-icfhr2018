
import cv2
import math
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HoughROIDetector:
    def __init__(self, rho, theta, min_intersections, min_line_length, max_line_gap, diag_threshold, n_workers):
        self.rho = rho
        self.theta = theta
        self.min_intersections = min_intersections
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        self.diag_threshold = diag_threshold

        self.n_workers = n_workers

    @staticmethod
    def separate_lines(hough_lines, diag_threshold):
        n_lines = hough_lines.shape[0]

        horizontal = []
        vertical = []
        diagonal = []

        for line in hough_lines:
            x1, y1, x2, y2 = line[0]

            if abs(x1 - x2) >= abs(y1 - y2):
                # check if horizontal ...
                angle = math.atan2(abs(y2 - y1), abs(x2 -x1))
                if angle < diag_threshold:
                    horizontal.append(line[0])
                else:
                    diagonal.append(line[0])
            else:
                # check if vertical ...
                angle = math.atan2(abs(x2 - x1), abs(y2 -y1))
                if angle < diag_threshold:
                    vertical.append(line[0])
                else:
                    diagonal.append(line[0])

        return horizontal, vertical, diagonal

    @staticmethod
    def line_intersection(line_1, line_2, extrapolate):
        l_x1, l_y1, l_x2, l_y2 = line_1
        s_x1, s_y1, s_x2, s_y2 = line_2

        if l_x1 == l_x2 and l_y1 == l_y2:
            #it's a point, no crossings with a single point...
            return None

        #check minimum/maximum
        l_xmin, l_xmax  = (l_x1, l_x2) if l_x1 < l_x2 else (l_x2, l_x1)
        l_ymin, l_ymax  = (l_y1, l_y2) if l_y1 < l_y2 else (l_y2, l_y1)

        if l_x1 != l_x2:
            # use the slope and intersect to compare...
            l_m = (l_y2 - l_y1) / (l_x2 - l_x1)
            l_b = l_y1 - l_m * l_x1

            if s_x2 == s_x1:
                # the segment is a vertical line...
                if extrapolate or (l_xmin <= s_x1 <= l_xmax):
                    # the vertical segment is inside the range of the current line...
                    y_int = s_x1 * l_m + l_b

                    # check if y_int in range of vertical line...
                    if extrapolate or (min(s_y1, s_y2) <= y_int <= max(s_y1, s_y2)):
                        # intersection found...
                        return s_x1, y_int
            else:
                # the segment is not a vertical line
                s_m = (s_y2 - s_y1) / (s_x2 - s_x1)
                s_b = s_y1 - s_m * s_x1

                # check if parallel
                if s_m == l_m:
                    # parallel lines, can only intersect if l_b == s_b
                    # (meaning they are the same line), and have intersecting ranges
                    if l_b == s_b:
                        if extrapolate or (l_xmin <= max(s_x1, s_x2) and min(s_x1, s_x2) <= l_xmax):
                             # intersection found, use middle point...
                             return (s_x1 + s_x2) / 2.0, (s_y1 + s_y2) / 2.0

                else:
                    # not parallel, they must have an intersection point
                    x_int = (s_b - l_b) / (l_m - s_m)
                    y_int = x_int * l_m + l_b

                    # the intersection point must be in both lines...
                    if extrapolate or ((l_xmin <= x_int <= l_xmax) and (min(s_x1, s_x2) <= x_int <= max(s_x1, s_x2))):
                        return (x_int, y_int)

        else:
            # the given line is a vertical line...
            # can't use the slope, use a different method
            if s_x2 == s_x1:
                # the segment is a vertical line (too)...
                # only if they are on the same x position, and their range intersects
                if s_x1 == l_x1 and (extrapolate or min(s_y1, s_y2) < l_ymax and l_ymin < max(s_y1, s_y2)):
                    return (s_x1 + s_x2) / 2.0, (s_y1 + s_y2) / 2.0
            else:
                # calculate intersection point
                if extrapolate or (min(s_x1, s_x2) <= l_x1 <= max(s_x1, s_x2)):
                    # the vertical line is inside the range of the current segment...
                    s_m = (s_y2 - s_y1) / (s_x2 - s_x1)
                    s_b = s_y1 - s_m * s_x1

                    y_int = l_x1 * s_m + s_b
                    # check if y_int in range of vertical line...
                    if extrapolate or (l_ymin <= y_int <= l_ymax):
                        # intersection found...
                        return l_x1, y_int

        return None

    @staticmethod
    def coord_to_int(coordinate):
        x, y = coordinate
        return int(round(x)), int(round(y))

    @staticmethod
    def get_quadrangle_lines(horizontal, vertical, width, height):
        # make all lines cut the image rectangle in two ...
        top_lines = []
        top_original = []
        bottom_lines = []
        bottom_original = []
        ver_mid_point = height / 2.0
        for x1, y1, x2, y2 in horizontal:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            y0 = int(round(b))
            y3 = int(round(m * width + b))

            mid_point = (y0 + y3) / 2.0
            if mid_point < ver_mid_point:
                top_lines.append((mid_point, [0, y0, width, y3]))
                top_original.append([x1, y1, x2, y2])
            else:
                bottom_lines.append((mid_point, [0, y0, width, y3]))
                bottom_original.append([x1, y1, x2, y2])

        left_lines = []
        left_original = []
        right_lines = []
        right_original = []
        hor_mid_point = width / 2.0
        for x1, y1, x2, y2 in vertical:
            m = (x2 - x1) / (y2 - y1)
            b = x1 - m * y1

            x0 = int(round(b))
            x3 = int(round(m * height + b))

            mid_point = (x0 + x3) / 2.0
            if mid_point < hor_mid_point:
                left_lines.append((mid_point, [x0, 0, x3, height]))
                left_original.append([x1, y1, x2, y2])
            else:
                right_lines.append((mid_point, [x0, 0, x3, height]))
                right_original.append([x1, y1, x2, y2])

        top_lines = [line for mid, line in sorted(top_lines, reverse=True)]
        bottom_lines = [line for mid, line in sorted(bottom_lines)]
        left_lines = [line for mid, line in sorted(left_lines, reverse=True)]
        right_lines = [line for mid, line in sorted(right_lines)]

        expanded = (top_lines, bottom_lines, left_lines, right_lines)
        original = (top_original, bottom_original, left_original, right_original)

        return expanded, original

    @staticmethod
    def get_line_image(video_image, original, expanded, diag):
        top, bottom, left, right = expanded
        o_top, o_bottom, o_left, o_right = original

        line_image = video_image.copy()

        for x1,y1,x2,y2 in left:
            cv2.line(line_image,(x1,y1),(x2,y2),(128,0,0),1)

        for x1,y1,x2,y2 in right:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,128,0),1)

        for x1,y1,x2,y2 in top:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,128),1)

        for x1,y1,x2,y2 in bottom:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,128,128),1)

        for x1,y1,x2,y2 in o_left:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)

        for x1,y1,x2,y2 in o_right:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)

        for x1,y1,x2,y2 in o_top:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),2)

        for x1,y1,x2,y2 in o_bottom:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,255),2)

        for x1,y1,x2,y2 in diag:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,255),2)

        return line_image

    @staticmethod
    def get_quadrilateral_image(h, w, top_edge, bottom_edge, left_edge, right_edge):
        # compute the quadrilateral defined by these edges ...
        top_left = HoughROIDetector.coord_to_int(HoughROIDetector.line_intersection(top_edge, left_edge, True))
        top_right = HoughROIDetector.coord_to_int(HoughROIDetector.line_intersection(top_edge, right_edge, True))
        bottom_left = HoughROIDetector.coord_to_int(HoughROIDetector.line_intersection(bottom_edge, left_edge, True))
        bottom_right = HoughROIDetector.coord_to_int(HoughROIDetector.line_intersection(bottom_edge, right_edge, True))

        polygon = np.array([top_left, top_right, bottom_right, bottom_left])

        polygon_image = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(polygon_image, polygon, (255,))

        return polygon_image


    def get_ROI(self, video_image, high_conf_bg, high_conf_fg, greedy, verbose):
        height, width = high_conf_bg.shape

        # Use hough transform to detect lines in the high conf. bg image
        lines = cv2.HoughLinesP(high_conf_bg, self.rho, self.theta, self.min_intersections,
                                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)

        # now, separate the lines into horizontal, vertical or diagonals (will be ignored)...
        hor, ver, diag = self.separate_lines(lines, self.diag_threshold)

        expanded, original = HoughROIDetector.get_quadrangle_lines(hor, ver, width, height)
        top, bottom, left, right = expanded

        # add boundaries ...
        top.append([0, 0, width, 0])
        bottom.append([0, height, width, height])
        left.append([0, 0, 0, height])
        right.append([width, 0, width, height])

        if verbose:
            print("Total Horizontal: " + str(len(hor) + 2))
            print("\tTop: " + str(len(top)))
            print("\tBottom: " + str(len(bottom)))
            print("")
            print("Total Vertical: " + str(len(ver) + 2))
            print("\tLeft: " + str(len(left)))
            print("\tRight: " + str(len(right)))
            print("Total Diagonal: " + str(len(diag)))

        # Test all boxes instead and select best
        # box_edges, box_polygon, box_score
        if greedy:
            # use greedy expansion (fast but produces sub-optimal boxes)
            box = HoughROIDetector.greedy_box_selection(video_image, high_conf_bg, high_conf_fg, top, bottom, left,
                                                        right, verbose, False)
        else:
            box = HoughROIDetector.box_selection(video_image, high_conf_bg, high_conf_fg, top, bottom, left, right,
                                                 True, self.n_workers)

        line_image = HoughROIDetector.get_line_image(video_image, original, expanded, diag)
        ROI_image = HoughROIDetector.get_ROI_image(video_image, box[1], box[0])

        debug_images = (line_image, ROI_image)

        return box, debug_images

    @staticmethod
    def evaluate_box_batch(batch_data):
        input_image, high_bg, high_fg, batch_boxes = batch_data

        batch_results = []
        for top_edge, bottom_edge, left_edge, right_edge in batch_boxes:
            box_score, box_polygon = HoughROIDetector.score_region(high_bg, high_fg, top_edge, bottom_edge, left_edge, right_edge)
            batch_results.append((box_score, top_edge, bottom_edge, left_edge, right_edge))

        return batch_results

    @staticmethod
    def box_selection(video_image, high_bg, high_fg, top, bottom, left, right, verbose, n_workers, batch_size=100):
        raw_boxes = list(itertools.product(top, bottom, left, right))

        n_batches = int(math.ceil(len(raw_boxes) / batch_size))
        batches = [(video_image, high_bg, high_fg, raw_boxes[batch_size * idx:batch_size * (idx + 1)]) for idx in range(n_batches)]

        all_boxes = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, batch_result in enumerate(executor.map(HoughROIDetector.evaluate_box_batch, batches)):
                all_boxes += batch_result

                if verbose:
                    print("Testing ... {0:d} out of {1:d}".format(len(all_boxes), len(raw_boxes)) + ("." * 20), end="\r")
        if verbose:
            print("")
            print("Total boxes tested: " + str(len(all_boxes)))

        all_boxes = sorted(all_boxes, key=lambda x:x[0], reverse=True)

        box_score, top_edge, bottom_edge, left_edge, right_edge = all_boxes[0]
        h, w, _ = video_image.shape
        box_polygon = HoughROIDetector.get_quadrilateral_image(h, w, top_edge, bottom_edge, left_edge, right_edge)
        box_edges = (top_edge, bottom_edge, left_edge, right_edge)

        return box_edges, box_polygon, box_score

    @staticmethod
    def greedy_box_selection(video_image, high_bg, high_fg, top, bottom, left, right, verbose, display_steps):
        left_edge = 0
        right_edge = 0
        top_edge = 0
        bottom_edge = 0
        best_polygon_edges = (top[top_edge], bottom[bottom_edge], left[left_edge], right[right_edge])
        best_score, best_polygon = HoughROIDetector.score_region(high_bg, high_fg, top[top_edge], bottom[bottom_edge],
                                                                 left[left_edge], right[right_edge])

        # try potential expansions, keep the best or stop if nothing improves
        expanded = True
        it_count = 0
        while expanded:
            expanded = False
            it_count += 1

            # try expanding on each direction (only one at a time)
            # and keep the best expansion if any improves the current best accuracy
            current_expansions = []

            if top_edge + 1 < len(top):
                exp_score, exp_polygon = HoughROIDetector.score_region(high_bg, high_fg, top[top_edge + 1], bottom[bottom_edge], left[left_edge], right[right_edge])
                current_expansions.append((exp_score, exp_polygon, 1, 0, 0, 0))

            if bottom_edge + 1 < len(bottom):
                exp_score, exp_polygon = HoughROIDetector.score_region(high_bg, high_fg, top[top_edge], bottom[bottom_edge + 1], left[left_edge], right[right_edge])
                current_expansions.append((exp_score, exp_polygon, 0, 1, 0, 0))

            if left_edge + 1 < len(left):
                exp_score, exp_polygon = HoughROIDetector.score_region(high_bg, high_fg, top[top_edge], bottom[bottom_edge], left[left_edge + 1], right[right_edge])
                current_expansions.append((exp_score, exp_polygon, 0, 0, 1, 0))

            if right_edge + 1 < len(right):
                exp_score, exp_polygon = HoughROIDetector.score_region(high_bg, high_fg, top[top_edge], bottom[bottom_edge], left[left_edge], right[right_edge + 1])
                current_expansions.append((exp_score, exp_polygon, 0, 0, 0, 1))

            if len(current_expansions) > 0:
                current_expansions = sorted(current_expansions, key=lambda x:x[0], reverse=True)

                if verbose:
                    print("Iteration #{0:d}, Initial best = {1:.4f}".format(it_count, best_score))
                    print([(exp_accuracy, exp_top, exp_bottom, exp_left, exp_right) for exp_accuracy, exp_polygon, exp_top, exp_bottom, exp_left, exp_right in current_expansions])

                best_expansion = current_expansions[0]
                exp_score, exp_polygon, exp_top, exp_bottom, exp_left, exp_right = best_expansion

                if exp_score > best_score:
                    # expand ...
                    expanded = True

                    # boundaries...
                    top_edge += exp_top
                    bottom_edge += exp_bottom
                    left_edge += exp_left
                    right_edge += exp_right

                    best_score = exp_score
                    best_polygon = exp_polygon
                    best_polygon_edges = (top[top_edge], bottom[bottom_edge], left[left_edge], right[right_edge])

                if display_steps:
                    ROI_image = HoughROIDetector.get_ROI_image(video_image, best_polygon, best_polygon_edges)
                    cv2.imshow("current ROI", ROI_image)
                    cv2.waitKey()

        return best_polygon_edges, best_polygon, best_score

    @staticmethod
    def get_ROI_image(input_image, board_mask, board_edges):
        ROI_image = input_image.copy()
        ROI_image = np.divide(ROI_image, 2).astype(np.uint8)
        ROI_image[board_mask > 0, :] = input_image[board_mask > 0, :].copy()

        top_edge, bottom_edge, left_edge, right_edge = board_edges

        x1,y1,x2,y2 = left_edge
        cv2.line(ROI_image,(x1,y1),(x2,y2),(255,0,0),2)

        x1,y1,x2,y2 = right_edge
        cv2.line(ROI_image,(x1,y1),(x2,y2),(0,255,0),2)

        x1,y1,x2,y2 = top_edge
        cv2.line(ROI_image,(x1,y1),(x2,y2),(0,0,255),2)

        x1,y1,x2,y2 = bottom_edge
        cv2.line(ROI_image,(x1,y1),(x2,y2),(0,255,255),2)

        return ROI_image

    @staticmethod
    def score_region(high_background, high_foreground, top_edge, bottom_edge, left_edge, right_edge):
        h, w  = high_background.shape

        polygon_image = HoughROIDetector.get_quadrilateral_image(h, w, top_edge, bottom_edge, left_edge, right_edge)

        fg_area = polygon_image.sum() / 255
        bg_area = h * w - fg_area

        total_high_bg = high_background.sum() / 255
        total_high_fg = high_foreground.sum() / 255

        polygon_mask = polygon_image > 0

        true_fg = high_foreground[polygon_mask].sum() / 255
        false_bg = high_background[polygon_mask].sum() / 255

        # FG correct: true_fg / total_high_fg
        # BG correct: (total_high_bg - false_bg) / total_high_bg
        # accuracy = (true_fg + (total_high_bg - false_bg)) / (total_high_bg + total_high_fg)
        if total_high_fg > 0:
            fg_accuracy = true_fg / total_high_fg
        else:
            fg_accuracy = 1.0

        if total_high_bg > 0:
            bg_accuracy = (total_high_bg - false_bg) / total_high_bg
        else:
            bg_accuracy = 1.0


        # prefers both high accuracy and large polygon area
        # score = accuracy * fg_area
        # score = fg_accuracy * fg_area + bg_accuracy * bg_area # Whatever has more pixels will dominate
        # score = (fg_accuracy * 0.5 + bg_accuracy * 0.5) # Loses some recall
        # score = ((2 * fg_accuracy * bg_accuracy) / (fg_accuracy + bg_accuracy)) # Looses recall!!!

        # Harmonic mean of foreground and background pixels accuracy by the foreground area
        score = ((2 * fg_accuracy * bg_accuracy) / (fg_accuracy + bg_accuracy)) * fg_area

        return score, polygon_image



