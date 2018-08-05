#==================================================
#  Class that defines operations related to
#  alignment of whiteboard content from videos
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     June 2015
#
#==================================================

import cv2
import numpy as np

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.preprocessing.data.visual_alignment import VisualAlignment
from AccessMath.util.misc_helper import MiscHelper
from .binarizer import Binarizer
from .loader import Loader


class Aligner:
    #number of frames to test alignment....
    ALIGNMENT_SAMPLE = 25
    #SURF_THRESHOLD = 0.3
    SURF_THRESHOLD = 0.4

    @staticmethod
    def computeTranslationAlignment(first_content, second_content, max_window, content_lum=255, sort_by=0):
        # assert that both images have the same dimensions
        assert len(first_content.shape) == 2
        assert len(second_content.shape) == 2
        assert first_content.shape == second_content.shape

        h, w = first_content.shape

        total_first = np.count_nonzero(first_content == content_lum)
        total_second = np.count_nonzero(second_content == content_lum)

        if total_first == 0 or total_second == 0:
            # cannot align, at least one of them is empty
            return 0.0, 0.0, 0.0, 0, 0

        alignment_results = []
        for disp_y in range(-max_window, max_window + 1):
            f_min_y = max(0, disp_y)
            f_max_y = h + min(0, disp_y)

            s_min_y = max(0, -disp_y)
            s_max_y = h + min(0, -disp_y)

            for disp_x in range(-max_window, max_window + 1):
                f_min_x = max(0, disp_x)
                f_max_x = w + min(0, disp_x)

                s_min_x = max(0, -disp_x)
                s_max_x = w + min(0, -disp_x)

                cut_first = first_content[f_min_y:f_max_y, f_min_x:f_max_x]
                cut_second = second_content[s_min_y:s_max_y, s_min_x:s_max_x]

                matches = np.count_nonzero(np.logical_and(cut_first == cut_second, cut_first == content_lum))

                recall = matches / total_first
                precision = matches / total_second

                if recall + precision > 0:
                    f_score = (2 * recall * precision) / (recall + precision)
                else:
                    f_score = 0

                # tempo_overlap = np.zeros((cut_first.shape[0], cut_first.shape[1], 3), dtype=np.uint8)
                # tempo_overlap[:, :, 0] = cut_first.copy()
                # tempo_overlap[:, :, 1] = cut_second.copy()

                # same_mask = tempo_overlap[:, :, 0] == tempo_overlap[:, :, 1]
                # tempo_overlap[same_mask, 2] = tempo_overlap[same_mask, 0]

                # alignment_results.append((f_score, recall, precision, disp_y, disp_x, tempo_overlap))
                alignment_results.append((f_score, recall, precision, disp_y, disp_x))

        alignment_results = sorted(alignment_results, reverse=True, key=lambda x:x[sort_by])

        return alignment_results[0]

    #======================================================================
    #  function that finds the equivalence of areas of content
    #  between videos (matches Content Regions) based on a set of
    #  keyframes
    #======================================================================
    @staticmethod
    def computeVisualAlignment(m_videos, a_videos, time_offset, motionless, save_frames, extraction_method_id):
        #distribute the selection of motionless frames...
        selected = MiscHelper.distribute_values( Aligner.ALIGNMENT_SAMPLE, 0, len(motionless) - 1 )

        #create the list..
        frame_list = []
        for idx in selected:
            frame_list.append( motionless[idx] )

        #extract the motionless frames from main videos
        frames = Loader.extractFramesRelative(m_videos, frame_list)
        if save_frames:
            for idx, f in enumerate(frames):
                abs_time, frame = f
                cv2.imwrite("out/main_" + str(idx) + ".jpg", frame)

        #calculate the absolute time for the corresponding frames
        #on the auxiliar video. Consider the time difference between videos
        times = [ (abs_time - time_offset) for abs_time, frame in frames ]

        #extract the motionless frames from auxiliar videos
        aux_frames = Loader.extractFramesAbsolute(a_videos, times)
        if save_frames:
            for idx, frame in enumerate(aux_frames):
                cv2.imwrite("out/auxiliar_" + str(idx) + ".jpg", frame)

        #find the visual correspondence between pairs of key frames
        matches_aux = []
        matches_main = []
        aux_boxes = []
        main_boxes = []

        all_content_main = []
        all_content_aux = []
        #...first... extract the content from each pair of frames...
        for i in range(min(Aligner.ALIGNMENT_SAMPLE, len(frames))):

            #get the current key frames
            abs_time, frame_main = frames[i]
            frame_aux = aux_frames[i]

            print( "Extracting content #" + str(i + 1) + " ... (Main: " +
                   TimeHelper.stampToStr(abs_time) + " - Aux: " +
                   TimeHelper.stampToStr(times[i]) + ")")

            #from the main key frame, extract content on the board
            main_box, content_main = Binarizer.frameContentBinarization(frame_main, extraction_method_id)
            main_boxes.append(main_box )

            #from the auxiliary key frame, extract content on the board
            aux_box, content_aux = Binarizer.frameContentBinarization(frame_aux, extraction_method_id)
            aux_boxes.append( aux_box )

            #add to list...
            all_content_main.append(content_main)
            all_content_aux.append(content_aux)

        #...then, extract the alignment.... keep highest score...
        all_scores  = []
        for i in range(min(Aligner.ALIGNMENT_SAMPLE, len(frames))):
            print( "Testing Alignment #" + str(i + 1) + " ... ")

            #corresponding frames....
            content_aux = all_content_aux[i]
            content_main = all_content_main[i]

            #Extract a set of good matches between these two images....
            # where object = aux content from mimio, to align with main content
            #       scene = main content to which the change regions will be projected
            aux_list, main_list = VisualAlignment.getSURFMatchingPoints(content_aux, content_main,
                                                                        Aligner.SURF_THRESHOLD)

            #generate projection based on these points...
            current_projection, mask = VisualAlignment.generateProjection( aux_list, main_list )
            #calculate score...
            score = VisualAlignment.getProjectionScore( current_projection, all_content_main, all_content_aux )

            #print( str(i) + " => " + str(score) )
            all_scores.append( (score, i, current_projection) )

            #add to the total list of points...
            matches_aux.append( aux_list )
            matches_main.append( main_list )

            #print( "ON " + str(i) + " where found " +  str(len(aux_list) ) + " matches" )

        all_scores = sorted( all_scores , reverse=True )

        #current best projection is the one with the top score...
        max_score = all_scores[0][0]
        all_matches_aux = matches_aux[all_scores[0][1]]
        all_matches_main = matches_main[all_scores[0][1]]
        best_projection = all_scores[0][2]

        #now, try to improve the quality of the projection by adding some keypoints from
        #candidate alignments with high scores and computing a new combined projection
        #for the list of combined keypoint matches...
        new_score = max_score
        pos = 1
        while new_score >= max_score and pos < len(all_scores):
            #add keypoints to the combined list...
            current_aux =  all_matches_aux + matches_aux[all_scores[pos][1]]
            current_main = all_matches_main + matches_main[all_scores[pos][1]]

            #generate the new projection...
            current_projection, mask = VisualAlignment.generateProjection( current_aux, current_main )

            #get score for combined projection...
            new_score = VisualAlignment.getProjectionScore( current_projection, all_content_main, all_content_aux )

            #check if score improved...
            if new_score >= max_score:
                #new best projection found....
                max_score  = new_score
                all_matches_aux += aux_list[all_scores[pos][1]]
                all_matches_main += main_list[all_scores[pos][1]]

                best_projection = current_projection
                pos += 1

        #Get the final alignment
        projection = best_projection

        print( "Best Alignment Score: " + str(max_score) )

        """
        # Un-comment to output alignment images
        for i in range(len(all_content_main)):
            content_main = all_content_main[i]
            content_aux = all_content_aux[i]

            proj_img = np.zeros( (content_main.shape[0], content_main.shape[1]), dtype=content_main.dtype )
            cv.WarpPerspective( cv.fromarray( content_aux ), cv.fromarray(proj_img), cv.fromarray( projection ) )

            result_image = np.zeros( (content_main.shape[0], content_main.shape[1], 3) )
            result_image[:,:,2] = content_main
            result_image[:,:,1] = proj_img

            #cv2.imshow('img',result_image)
            cv2.imwrite( 'DEBUG_MAIN_' + str(i) + '.bmp', content_main )
            cv2.imwrite( 'DEBUG_AUX_' + str(i) + '.bmp', content_aux )
            cv2.imwrite( 'DEBUG_PROJECTION_' + str(i) + '.bmp' , result_image )
        """

        #average of the boxes of the whiteboard
        main_box = MiscHelper.averageBoxes(main_boxes)
        aux_box = MiscHelper.averageBoxes(aux_boxes)

        #store them in a single object...
        visual_alignment = VisualAlignment()
        # ... main size...
        visual_alignment.main_width = frames[0][1].shape[1]
        visual_alignment.main_height = frames[0][1].shape[0]
        #.... main board box ...
        visual_alignment.main_box = main_box
        # ... aux size ....
        visual_alignment.aux_width = aux_frames[0].shape[1]
        visual_alignment.aux_height = aux_frames[0].shape[0]
        #... aux board box...
        visual_alignment.aux_box = aux_box
        #... projection ....
        visual_alignment.projection = projection

        return visual_alignment