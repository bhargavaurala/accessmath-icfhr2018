
# ============================================================
#  Interface functionality between AccessMath and Tangent-V
#
#  Created by: Kenny Davila
#              (March, 2018)
# ============================================================

from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.data.space_time_struct import SpaceTimeStruct
from AccessMath.preprocessing.config.parameters import Parameters

from TangentV.retrieval.graph_match_group import GraphMatchGroup


class TangentV_Helper:
    cache_3DSTs = {}
    PerLectureKeyFramesRefs = {}  # [ lecture_id] [ 0 ... N ] = (kf index, image_id, image_REF)
    PerLectureKeyFramesOffsets = {}  # [ lecture_id] [ image_id ] = offset
    VisualizerServer = None

    @staticmethod
    def prepare(args):
        # load database
        try:
            database = MetaDataDB.from_file(args['database'])
        except:
            print("Invalid AccessMath database file")
            return

        TangentV_Helper.VisualizerServer = database.indexing.visualization_server

        # ... Load 3D structures ....
        print("Loading CC indices per lecture ... ")
        for lecture in database.lectures:
            struct_filename = database.output_temporal + '/' + Parameters.Output_ST3D + str(lecture.id) + ".dat"
            TangentV_Helper.cache_3DSTs[lecture.title] = MiscHelper.dump_load(struct_filename)

    @staticmethod
    def load_cache_lecture_metadata(image_index, lecture_id):
        key_frames_images = image_index.get_images_by_attribute_value("lecture", lecture_id)

        offsets = {}    # [ image_id ] = offset
        references = [] # [ 0 ... N ] = (kf index, image_id, image_REF)
        for location_id, image_id, image_ref in key_frames_images:
            _, kf_idx = location_id.split(":")
            kf_idx = int(kf_idx)

            references.append((kf_idx, image_id, image_ref))

        references = sorted(references, key= lambda x:x[0])
        for offset, (kf_idx, image_id, image_ref) in enumerate(references):
            offsets[image_id] = offset

        TangentV_Helper.PerLectureKeyFramesRefs[lecture_id] = references
        TangentV_Helper.PerLectureKeyFramesOffsets[lecture_id] = offsets

    @staticmethod
    def generate_match_link(image_index, match_group):
        assert isinstance(match_group, GraphMatchGroup)

        # get match lecture key-frames (... if not in cache already ...)
        # ... first identify match lecture id ...
        any_image_id = int(list(match_group.matches_per_image.keys())[0])
        lecture_vals = image_index.get_image_attribute(any_image_id, "lecture")
        lecture_id = lecture_vals[0][1]

        # get cached spatio-temporal structure
        st3D = TangentV_Helper.cache_3DSTs[lecture_id]

        # if key-frames have not been loaded ...
        if not lecture_id in TangentV_Helper.PerLectureKeyFramesRefs:
            TangentV_Helper.load_cache_lecture_metadata(image_index, lecture_id)

        # get key-frame info ...
        lecture_kf_refs = TangentV_Helper.PerLectureKeyFramesRefs[lecture_id]
        lecture_kf_offs = TangentV_Helper.PerLectureKeyFramesOffsets[lecture_id]

        oldest_time = TangentV_Helper.best_match_start_time(match_group, st3D, lecture_kf_refs, lecture_kf_offs)
        str_time = "&t=" + str(int(oldest_time / 1000))

        match_link = TangentV_Helper.VisualizerServer + "?lecture=" + lecture_id + str_time

        return match_link


    @staticmethod
    def best_match_start_time(match_group, match_3DST, lecture_kf_refs, lecture_kf_offsets):
        assert isinstance(match_group, GraphMatchGroup)
        assert isinstance(match_3DST, SpaceTimeStruct)

        # sort first by score (higher score first) and then by negative keyframe index (smallest first)
        scored_matches = sorted([(match_group.matches_per_image[image_id].total_score, -image_id)
                                for image_id in match_group.matches_per_image], reverse=True)

        # use top scored match in the group ...
        _, top_image_id = scored_matches[0]
        top_image_id = -top_image_id
        top_match = match_group.matches_per_image[top_image_id]

        # find ...
        image_offset = lecture_kf_offsets[top_image_id]
        top_keyframe_idx, _, _ = lecture_kf_refs[image_offset]

        if image_offset == 0:
            start_frame_idx = 0
        else:
            start_frame_idx, _, _ = lecture_kf_refs[image_offset - 1]

        # start by filtering (reducing) candidate groups
        # ... on the same key-frame ...
        frame_groups = match_3DST.groups_in_frame_range(start_frame_idx, top_keyframe_idx)
        # ... on the same region ...
        t_min_x, t_max_x, t_min_y, t_max_y = match_group.regions_per_image[top_image_id]
        region_groups = match_3DST.groups_in_space_region(t_min_x, t_max_x, t_min_y, t_max_y, frame_groups)

        # identify exact matching groups (greedily)
        # match CC might include "Grouped" CC from the original structure
        match_ccs = [top_match.candidate_graph.vertices[cc_idx] for cc_idx in top_match.candidate_nodes]
        region_ccs = match_3DST.get_CC_instances(region_groups, top_keyframe_idx)

        # compute pair-wise overlaps pixel-level f-score
        match_overlaps = []
        for m_idx, m_cc in enumerate(match_ccs):
            # print(m_cc.getBoundingBox())
            for r_idx, r_cc in enumerate(region_ccs):
                iou_score = m_cc.getOverlapIOU(r_cc)
                match_overlaps.append((iou_score, m_idx, r_idx))

        # greedy choose 1-to-1 matches with highest overlap
        match_overlaps = sorted(match_overlaps, key=lambda x: x[0], reverse=True)
        region_matches = [None for m_cc in match_ccs]
        for score, m_idx, r_idx in match_overlaps:
            if region_matches[m_idx] is None:
                region_matches[m_idx] = region_groups[r_idx]

        # find the "oldest" matched cc
        oldest_idx, oldest_frame, oldest_time = match_3DST.find_oldest_in_group(region_matches)

        return oldest_time


def get_link_generator():
    return TangentV_Helper
