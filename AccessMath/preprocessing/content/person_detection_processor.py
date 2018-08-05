
import xml.etree.ElementTree as ET

import numpy as np

from AccessMath.annotation.video_object import VideoObject
from AccessMath.annotation.video_object_location import VideoObjectLocation

class PersonDetectionProcessor:
    def __init__(self, all_frame_ids, all_boxes, all_abs_times, all_visible):
        self.frame_ids = all_frame_ids
        self.abs_times = all_abs_times
        self.bboxes = all_boxes
        self.visible = all_visible

    def find_nonzero_intervals(self, non_zero):
        # this function finds the set of contiguous intervals on
        if non_zero.shape[0] == 0:
            return []

        intervals = []
        int_ini = 0

        for idx in range(1, non_zero.shape[0]):
            if non_zero[idx] - non_zero[idx - 1] > 1:
                # end of last interval found
                intervals.append((int_ini, idx - 1))

                # start new interval
                int_ini = idx

        intervals.append((int_ini, non_zero.shape[0] - 1))

        return intervals

    def interpolation_mse(self, key_frames, interp_x, y_values):
        interp_kf_x = np.flatnonzero(key_frames)
        interp_kf_y = y_values[key_frames]
        keyframe_count = interp_kf_x.shape[0]

        interp_y = np.interp(interp_x, interp_kf_x, interp_kf_y)

        # TODO: could be speed up if above can be speed up
        interp_sq_error = np.power(y_values - interp_y, 2)

        mse = interp_sq_error.mean()
        x_max_error = np.argmax(interp_sq_error)

        return keyframe_count, interp_y, mse, x_max_error

    def refine_keyframes(self, initial_keyframes, target_values, min_mse, max_frames):
        key_frames = initial_keyframes.copy()

        # interpolation axis ...
        interp_x = np.arange(0, target_values.shape[0])

        keyframe_count, interp_y, mse, x_max_error = self.interpolation_mse(key_frames, interp_x, target_values)

        while mse > min_mse and keyframe_count < max_frames:
            # add key-frame ...
            key_frames[x_max_error] = True
            # repeat ..
            keyframe_count, interp_y, mse, x_max_error = self.interpolation_mse(key_frames, interp_x, target_values)

        return key_frames, interp_y, keyframe_count, mse

    def identify_keyframes(self, verbose, debug_mark_all=False):
        n_frames = self.bboxes.shape[0]

        all_cx = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2.0
        all_cy = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2.0
        # all_w = (all_boxes[:, 2] - all_boxes[:, 0])
        # all_h = (all_boxes[:, 3] - all_boxes[:, 1])
        # all_area = (all_w * all_h)

        # use array to mark interpolation key-frames
        key_frames = np.zeros(n_frames, np.bool)

        # DEBUG: all are marked as key-frames
        if debug_mark_all:
            key_frames[:] = True
        else:
            # extremes are always key-frames
            key_frames[0] = True
            key_frames[-1] = True

            # ... adding the start and end of "invisible frames" as key-frames ...
            not_visible = np.flatnonzero(np.logical_not(self.visible))

            non_visible_int = self.find_nonzero_intervals(not_visible)
            for int_start, int_end in non_visible_int:
                idx_not_visible_start = not_visible[int_start]
                idx_not_visible_end = not_visible[int_end]

                # mark interval boundaries as key-frames
                key_frames[idx_not_visible_start] = True
                if idx_not_visible_end + 1 < n_frames:
                    key_frames[idx_not_visible_end + 1] = True

        cx_key_frames, interp_cx, count_kf_cx, cx_mse = self.refine_keyframes(key_frames, all_cx, 100, 5000)
        cy_key_frames, interp_cy, count_kf_cy, cy_mse = self.refine_keyframes(key_frames, all_cy, 100, 5000)

        combined_key_frames = np.logical_or(cx_key_frames, cy_key_frames)

        interp_x = np.arange(0, n_frames)
        _, _, cx_mse, _ = self.interpolation_mse(combined_key_frames, interp_x, all_cx)
        _, _, cy_mse, _ = self.interpolation_mse(combined_key_frames, interp_x, all_cy)

        final_keyframes_idxs = np.flatnonzero(combined_key_frames)

        if verbose:
            print("-> Final Center-X MSE: {0:.4f}".format(cx_mse))
            print("-> Final Center-Y MSE: {0:.4f}".format(cy_mse))

            msg = "-> Total key-frames identified: {0:d} (out of {1:d} frames)"
            print(msg.format(final_keyframes_idxs.shape[0], self.bboxes.shape[0]))

        return final_keyframes_idxs

    def add_speaker_to_annotations(self, input_main_file, output_filename, source_width, source_height, keyframes_idxs,
                                   verbose=True):
        # ... get element tree object ...
        annotation_tree = ET.parse(input_main_file)
        annotation_root = annotation_tree.getroot()

        objects_root = annotation_root.find("VideoObjects")
        for object_xml in objects_root.findall("VideoObject"):
            if object_xml.find("Name").text == "speaker":
                objects_root.remove(object_xml)
                if verbose:
                    print("-> Existing speaker data found and removed!")

        # offset = player pos - canvas pos
        drawing_root = annotation_root.find("DrawingInfo")
        canvas_root = drawing_root.find("Canvas")
        render_area_root = drawing_root.find("Player").find("RenderArea")
        canvas_x = float(canvas_root.find("X").text)
        canvas_y = float(canvas_root.find("Y").text)
        render_x = float(render_area_root.find("X").text)
        render_y = float(render_area_root.find("Y").text)
        render_w = float(render_area_root.find("W").text)
        render_h = float(render_area_root.find("H").text)

        offset_x = render_x - canvas_x
        offset_y = render_y - canvas_y
        scale_w = render_w / source_width
        scale_h = render_h / source_height

        if verbose:
            print("-> Generating speaker data ... ")

        # ... generate the speaker data
        speaker_object = VideoObject("speaker", "speaker")

        for keyframe_idx in keyframes_idxs:
            abs_frame = self.frame_ids[keyframe_idx]
            abs_time = self.abs_times[keyframe_idx]
            visible = self.visible[keyframe_idx]

            x1, y1, x2, y2 =  self.bboxes[keyframe_idx]
            x = offset_x + x1 * scale_w
            y = offset_y + y1 * scale_h
            w = (x2 - x1) * scale_w
            h = (y2 - y1) * scale_h

            # boxes in image space need to be converted to drawing canvas space
            # ... first scale by drawing canvas size (vs video size)
            speaker_object.set_location_at(abs_frame, abs_time, visible, x, y, w, h)

        if verbose:
            print("-> Saving annotation file ... ")

        # ... convert speaker data to XML ...
        speaker_xml_node = ET.fromstring(speaker_object.toXML())
        # ... add to the tree
        objects_root.append(speaker_xml_node)

        # save to output file
        annotation_tree.write(output_filename)


    @staticmethod
    def from_raw_info(frame_info):
        all_boxes = np.zeros((len(frame_info), 4), np.float64)
        all_visible = np.zeros(len(frame_info), np.bool)
        all_abs_times = np.zeros(len(frame_info), np.float64)
        all_frame_ids = sorted(list(frame_info.keys()))
        last_known_box_idx = 0

        # for each frame ... copy to numpy arrays ...
        for idx, frame_id in enumerate(all_frame_ids):
            current_frame = frame_info[frame_id]

            if len(current_frame["confidences"]) > 0:
                top_box = np.argmax(current_frame["confidences"])
            else:
                top_box = -1

            all_abs_times[idx] = current_frame["abs_time"]

            if len(current_frame["bboxes"]) == 1:
                # only one bounding box on the frame
                all_boxes[idx, :] = current_frame["bboxes"][top_box]
                all_visible[idx] = True
                last_known_box_idx = idx
            elif len(current_frame["bboxes"]) > 1:
                # multiple bounding boxes ...
                if idx == 0:
                    # on first frame ? use the most confident one ...
                    best_box = top_box
                else:
                    # find the box with the largest IOU compared to the previous frame box ...
                    pre_x1, pre_y1, pre_x2, pre_y2 = all_boxes[idx - 1, :]
                    pre_w = pre_x2 - pre_x1 + 1
                    pre_h = pre_y2 - pre_y1 + 1
                    pre_loc = VideoObjectLocation(True, 0, 0, pre_x1, pre_y1, pre_w, pre_h)

                    boxes_IOUs = []
                    for box_x1, box_y1, box_x2, box_y2 in current_frame["bboxes"]:
                        box_w = box_x2 - box_x1 + 1
                        box_h = box_y2 - box_y1 + 1
                        box_loc = VideoObjectLocation(True, 0, 0, box_x1, box_y1, box_w, box_h)

                        box_IOU = box_loc.IOU(pre_loc)
                        boxes_IOUs.append(box_IOU)

                    top_IOU = np.argmax(boxes_IOUs)

                    if boxes_IOUs[top_IOU] < 0.25:
                        # too low IOU, use top box ...
                        best_box = top_box
                    else:
                        # use the box with maximum IOU with previous position ...
                        best_box = top_IOU

                # confidences
                all_boxes[idx, :] = current_frame["bboxes"][best_box]
                all_visible[idx] = True
                last_known_box_idx = idx
            else:
                # no bounding boxes, copy from last previously known position ...
                all_boxes[idx, :] = all_boxes[last_known_box_idx, :]
                all_visible[idx] = False

        return PersonDetectionProcessor(all_frame_ids, all_boxes, all_abs_times, all_visible)



