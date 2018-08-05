
import numpy as np

from AccessMath.annotation.text_annotation_exporter import TextAnnotationExporter

class TextDetectionEvaluator:
    def __init__(self, min_confidence=None):
        self.min_confidence = min_confidence
        self.training_thresholds = np.arange(0.05, 1.0, 0.05)
        self.results_per_lecture = {}

    def get_text_det_metrics(self, text_detection, keyframe_gt, min_confidence=None):
        # sorted key-frames
        keyframe_ids = sorted(list(text_detection.keys()))

        # extract and group detected bounding boxes ...
        all_recall = []
        all_precision = []
        all_fscore = []
        all_gt_counts = []
        all_det_counts = []
        for keyframe_id in keyframe_ids:
            frame_results = text_detection[keyframe_id]

            # find ground truth frame ...
            gt_frame = keyframe_gt[keyframe_id]["pixel_visible"]
            gt_count = keyframe_gt[keyframe_id]["total_visible"]

            # generate boolean image for current frame bboxes ...
            det_frame = np.zeros(gt_frame.shape, np.bool)
            if min_confidence is None:
                confidences = None
            else:
                confidences = frame_results["confidences"]

            total_valid_boxes = 0
            for bbox_idx, (x1, y1, x2, y2) in enumerate(frame_results["bboxes"]):
                # mark ...
                if min_confidence is None or confidences[bbox_idx] >= min_confidence:
                    total_valid_boxes += 1
                    det_frame[int(y1):int(y2), int(x1):int(x2)] = True

            matched = np.logical_and(gt_frame, det_frame)
            pixels_matched = np.count_nonzero(matched)
            total_pixels_gt = np.count_nonzero(gt_frame)
            total_pixels_det = np.count_nonzero(det_frame)

            if total_pixels_gt > 0:
                recall = pixels_matched / total_pixels_gt
            else:
                recall = 1.0

            if total_pixels_det > 0:
                precision = pixels_matched / total_pixels_det
            else:
                precision = 1.0

            if recall + precision > 0.0:
                fscore = (2.0 * recall * precision) / (recall + precision)
            else:
                fscore = 0.0

            all_recall.append(recall)
            all_precision.append(precision)
            all_fscore.append(fscore)
            all_gt_counts.append(gt_count)
            all_det_counts.append(total_valid_boxes)

            # print((keyframe_id, recall, precision, fscore))

        detection_results = {
            "avg_recall": np.mean(all_recall),
            "avg_precision": np.mean(all_precision),
            "avg_fscore": np.mean(all_fscore),
            "avg_gt_count": np.mean(all_gt_counts),
            "avg_det_count": np.mean(all_det_counts),
        }

        return detection_results

    def generate_per_frame_gt(self, key_frame_ids, text_exporter):
        all_gt_frames = {}
        for frame_idx in key_frame_ids:
            # find speaker and visible bboxes ...
            speaker_loc, not_occluded_bboxes, occluded_bboxes = text_exporter.frame_visible_bboxes_state(frame_idx)

            # initially... nothing is visible ...
            gt_frame = np.zeros((text_exporter.img_height, text_exporter.img_width), np.bool)

            # for all non-occluded bboxes ..
            for x1, y1, x2, y2 in not_occluded_bboxes:
                # mark as text region ....
                gt_frame[int(y1):int(y2), int(x1):int(x2)] = True

            all_gt_frames[frame_idx] = {
                "total_occluded": len(occluded_bboxes),
                "total_visible": len(not_occluded_bboxes),
                "pixel_visible": gt_frame,
            }

        return all_gt_frames

    def process_input(self, process, input_data):
        # person_detection, raw_text_detection, refined_text_detection = input_data
        raw_text_detection, refined_text_detection = input_data

        # TODO: this should come from some of the detectors .... ?
        width, height = 1920, 1080

        # these shouldn't be tuples ...
        raw_text_detection = raw_text_detection[0]
        refined_text_detection = refined_text_detection[0]

        # all_frame_ids, spk_bboxes, all_abs_times, spk_visible = self.process_person_detection_results(person_detection)
        text_exporter = TextAnnotationExporter.FromAnnotationXML(process.database, process.current_lecture)
        text_exporter.initialize(width, height, False)

        # from text detection ...
        keyframe_ids = sorted(list(raw_text_detection.keys()))

        keyframe_gt = self.generate_per_frame_gt(keyframe_ids, text_exporter)

        print("-> Computing Raw Text Detection Metrics")
        raw_detection_metrics = self.get_text_det_metrics(raw_text_detection, keyframe_gt, self.min_confidence)
        print("-> Computing Ref. Text Detection Metrics")
        refined_detection_metrics = self.get_text_det_metrics(refined_text_detection, keyframe_gt, None)

        current_results = {
            "raw": raw_detection_metrics,
            "refined": refined_detection_metrics,
        }

        self.results_per_lecture[process.current_lecture.title] = current_results

    def process_train_input(self, process, input_data):
        raw_text_detection = input_data[0]

        # TODO: this should come from some of the detectors .... ?
        width, height = 1920, 1080

        # all_frame_ids, spk_bboxes, all_abs_times, spk_visible = self.process_person_detection_results(person_detection)
        text_exporter = TextAnnotationExporter.FromAnnotationXML(TextAnnotationExporter.ExportModeAllPerFrame,
                                                                 process.database, process.current_lecture,
                                                                 None)
        text_exporter.initialize(width, height, False)

        # from text detection ...
        keyframe_ids = sorted(list(raw_text_detection.keys()))

        keyframe_gt = self.generate_per_frame_gt(keyframe_ids, text_exporter)

        print("Evaluating confidence thresholds for Lecture: " + process.current_lecture.title)
        print("\nTh\tBoxes\tRec.\tPrec.\tF.Score")
        row_str = "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}"

        current_results = {}
        for conf_threshold in self.training_thresholds:
            metrics = self.get_text_det_metrics(raw_text_detection, keyframe_gt, conf_threshold)

            current_results[conf_threshold] = metrics
            print(row_str.format(conf_threshold, metrics["avg_det_count"], metrics["avg_recall"],
                                 metrics["avg_precision"], metrics["avg_fscore"]))

        self.results_per_lecture[process.current_lecture.title] = current_results
        print("")

    def print_totals(self):
        print(" \t \tRaw\t \t \t \tRefined")
        print("Lectures\tGT Box\tBoxes\tRec.\tPrec.\tF.Score\tBoxes\tRec.\tPrec.\tF.Score")

        # prepare ...
        all_raw_recall, all_raw_precision, all_raw_fscore, all_raw_counts = [], [], [], []
        all_ref_recall, all_ref_precision, all_ref_fscore, all_ref_counts = [], [], [], []
        all_gt_counts = []

        row = "{0:s}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}"
        for lecture_id in sorted(list(self.results_per_lecture.keys())):
            lecture_res = self.results_per_lecture[lecture_id]
            raw = lecture_res["raw"]
            ref = lecture_res["refined"]

            all_gt_counts.append(raw["avg_gt_count"])

            all_raw_recall.append(raw["avg_recall"])
            all_raw_precision.append(raw["avg_precision"])
            all_raw_fscore.append(raw["avg_fscore"])
            all_raw_counts.append(raw["avg_det_count"])

            all_ref_recall.append(ref["avg_recall"])
            all_ref_precision.append(ref["avg_precision"])
            all_ref_fscore.append(ref["avg_fscore"])
            all_ref_counts.append(ref["avg_det_count"])

            print(row.format(lecture_id, raw["avg_gt_count"], raw["avg_det_count"], raw["avg_recall"] * 100.0,
                             raw["avg_precision"] * 100.0, raw["avg_fscore"] * 100.0, ref["avg_det_count"],
                             ref["avg_recall"] * 100.0, ref["avg_precision"] * 100.0, ref["avg_fscore"] * 100.0))

        avg_gt_count = np.mean(all_gt_counts)

        raw_avg_recall = np.mean(all_raw_recall)
        raw_avg_precision = np.mean(all_raw_precision)
        raw_avg_fscore = np.mean(all_raw_fscore)
        raw_avg_count = np.mean(all_raw_counts)

        ref_avg_recall = np.mean(all_ref_recall)
        ref_avg_precision = np.mean(all_ref_precision)
        ref_avg_fscore = np.mean(all_ref_fscore)
        ref_avg_count = np.mean(all_ref_counts)

        print(row.format("Averages", avg_gt_count, raw_avg_count, raw_avg_recall * 100.0, raw_avg_precision * 100.0,
                         raw_avg_fscore * 100.0, ref_avg_count, ref_avg_recall * 100.0, ref_avg_precision * 100.0,
                         ref_avg_fscore * 100.0))

    def print_train_totals(self):
        print("\nSummary of confidence thresholds\n")
        print("\nTh\tGT Box\tBoxes\tRec.\tPrec.\tF.Score")

        # for each threshold ...
        row_str = "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}"
        all_mean_fscores = []
        for conf_threshold in self.training_thresholds:
            all_raw_recall, all_raw_precision, all_raw_fscore, all_raw_counts = [], [], [], []
            all_gt_counts = []

            for lecture_id in sorted(list(self.results_per_lecture.keys())):
                lecture_res = self.results_per_lecture[lecture_id]

                raw = lecture_res[conf_threshold]

                all_gt_counts.append(raw["avg_gt_count"])

                all_raw_recall.append(raw["avg_recall"])
                all_raw_precision.append(raw["avg_precision"])
                all_raw_fscore.append(raw["avg_fscore"])
                all_raw_counts.append(raw["avg_det_count"])

            avg_gt_count = np.mean(all_gt_counts)

            raw_avg_recall = np.mean(all_raw_recall)
            raw_avg_precision = np.mean(all_raw_precision)
            raw_avg_fscore = np.mean(all_raw_fscore)
            raw_avg_count = np.mean(all_raw_counts)

            all_mean_fscores.append(raw_avg_fscore)

            print(row_str.format(conf_threshold, avg_gt_count, raw_avg_count, raw_avg_recall, raw_avg_precision,
                                 raw_avg_fscore))

        best_score_idx = np.argmax(all_mean_fscores)
        print("\nBest threshold: " + str(self.training_thresholds[best_score_idx]))
