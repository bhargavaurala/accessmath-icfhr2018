
import time

import numpy as np
import cv2

from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.unique_cc_group import UniqueCCGroup

from .evaluator import Evaluator
from .eval_parameters import EvalParameters

class SummaryEvaluator:
    def __init__(self):
        self.per_lecture_metrics = {}
        self.total_per_lecture_keyframes = {}
        self.ranges_per_lecture = {}
        self.range_names = self.__get_sorted_size_ranges_names__()

    def __get_sorted_size_ranges_names__(self):
        range_boundaries = [0.0]
        for percentile in EvalParameters.UniqueCC_size_percentiles:
            range_boundaries.append(percentile)
        range_boundaries.append(100.0)

        range_names = []
        for idx in range(len(EvalParameters.UniqueCC_size_percentiles) + 1):
            name = "{0:.2f}% to {1:.2f}%".format(range_boundaries[idx], range_boundaries[idx + 1])
            range_names.append(name)

        range_names.append("all")

        return range_names

    def process_summary(self, process, input_data):
        database = process.database
        lecture = process.current_lecture

        if "b" in process.params:
            base_line_prefix = process.params["b"] + "_"
        else:
            base_line_prefix = ""

        lecture_suffix = database.name + "_" + lecture.title.lower()

        summary_prefix = database.output_summaries + "/" + base_line_prefix + lecture_suffix
        annotation_prefix = database.output_annotations + "/" + lecture_suffix

        annot_filename = annotation_prefix + "/segments.xml"
        annot_cc_groups_filename = annotation_prefix + "/unique_ccs.xml"
        annot_image_prefix = annotation_prefix + "/keyframes/"
        annot_binary_prefix = annotation_prefix + "/binary/"

        print("-> loading data ...")
        start_loading = time.time()

        # ideal summary ...
        annot_keyframes, annot_segments = KeyFrameAnnotation.LoadExportedKeyframes(annot_filename, annot_image_prefix,
                                                                                   True)
        for keyframe in annot_keyframes:
            keyframe.binary_image = cv2.imread(annot_binary_prefix + str(keyframe.idx) + ".png")
            keyframe.update_binary_cc(False)

        annot_keyframes = KeyFrameAnnotation.CombineKeyframesPerSegment(annot_keyframes, annot_segments, False)

        annot_cc_group, annot_unique_groups = UniqueCCGroup.GroupsFromXML(annot_keyframes, annot_cc_groups_filename)

        # provided summary ...
        summ_filename = summary_prefix + "/segments.xml"
        summ_image_prefix = summary_prefix + "/keyframes/"
        summ_keyframes, summ_segments = KeyFrameAnnotation.LoadExportedKeyframes(summ_filename, summ_image_prefix, True,
                                                                                 False, True)
        for keyframe in summ_keyframes:
            # keyframe.binary_image = keyframe.raw_image.copy()
            keyframe.update_binary_cc(False)

        summ_keyframes = KeyFrameAnnotation.CombineKeyframesPerSegment(summ_keyframes, summ_segments, False)

        print("-> data loaded!")
        print("-> computing metrics ...")

        # TODO: This should come from configuration
        # output_prefix = database.output_evaluation + "/" + base_line_prefix + database.name + "_" + lecture.title.lower()
        output_prefix = "output/evaluation" + "/" + base_line_prefix + lecture_suffix

        # compute metrics and store matching results as images ...
        EvalParameters.Report_Summary_Show_stats_per_size = True
        all_metrics, ranges = Evaluator.compute_summary_metrics(annot_segments, annot_keyframes, annot_unique_groups,
                                                                annot_cc_group, summ_segments, summ_keyframes, False,
                                                                output_prefix)

        self.per_lecture_metrics[lecture.title] = all_metrics
        self.total_per_lecture_keyframes[lecture.title] = len(summ_keyframes)
        self.ranges_per_lecture[lecture.title] = ranges

    def basic_totals_per_minRP(self):
        stats_per_minRP = {}

        for lecture_id in self.per_lecture_metrics:
            # capture metrics for all ranges ...

            for range in self.per_lecture_metrics[lecture_id]:
                lecture_metrics = self.per_lecture_metrics[lecture_id][range]

                if range == "all":
                    range_name = "all"
                else:
                    # location of range name in sorted ranges ...
                    range_idx = self.ranges_per_lecture[lecture_id].index(range)
                    range_name = self.range_names[range_idx]

                # for each min-R, min-P
                for match_level_metrics in lecture_metrics:
                    level_key = "{0:.2f}\t{1:.2f}".format(match_level_metrics["min_cc_recall"] * 100.0,
                                                          match_level_metrics["min_cc_precision"] * 100.0)

                    if not level_key in stats_per_minRP:
                        stats_per_minRP[level_key] = {}

                    if not range_name in stats_per_minRP[level_key]:
                        stats_per_minRP[level_key][range_name] = {}

                    recall_metrics = match_level_metrics["recall_metrics"]
                    precision_metrics = match_level_metrics["precision_metrics"]

                    stats_per_minRP[level_key][range_name][lecture_id] = {
                        "global": {
                            "recall": recall_metrics["recall"] * 100.0,
                            "precision": precision_metrics["precision"] * 100.0,
                        },
                        "per_frame": {
                            "recall": recall_metrics["avg_recall"] * 100.0,
                            "precision": precision_metrics["avg_precision"] * 100.0,
                        }
                    }

        return stats_per_minRP

    def print_totals(self, all_ranges=False):
        title = " \t \tGlob.\t \tAVG\t"
        sub_title = "Lect.\tFrames\tRec.\tPrec.\tRec.\tPrec."
        row = "{0:s}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}"

        stats_per_minRP = self.basic_totals_per_minRP()

        # for each min-R, min-P
        for level_key in sorted(list(stats_per_minRP.keys())):
            level_stats = stats_per_minRP[level_key]

            print("\n" + "=" * 50)
            print("Min CC recall - Min CC precision: " + level_key)

            if all_ranges:
                current_ranges = self.range_names
            else:
                current_ranges = ["all"]

            for range in current_ranges:
                range_stats = level_stats[range]

                print("\nSize Range: " + range)
                print(title)
                print(sub_title)

                lvl_counts, lvl_recall, lvl_precision, lvl_avg_recall, lvl_avg_precision = [], [], [], [], []
                for lecture_id in sorted(list(range_stats.keys())):
                    lecture_stats = range_stats[lecture_id]

                    lvl_counts.append(self.total_per_lecture_keyframes[lecture_id])
                    lvl_recall.append(lecture_stats["global"]["recall"])
                    lvl_precision.append(lecture_stats["global"]["precision"])
                    lvl_avg_recall.append(lecture_stats["per_frame"]["recall"])
                    lvl_avg_precision.append(lecture_stats["per_frame"]["precision"])

                    print(row.format(lecture_id, self.total_per_lecture_keyframes[lecture_id],
                                     lecture_stats["global"]["recall"], lecture_stats["global"]["precision"],
                                     lecture_stats["per_frame"]["recall"], lecture_stats["per_frame"]["precision"]))

                print(row.format("Averages", np.mean(lvl_counts), np.mean(lvl_recall), np.mean(lvl_precision),
                                 np.mean(lvl_avg_recall), np.mean(lvl_avg_precision)))
                print("")

