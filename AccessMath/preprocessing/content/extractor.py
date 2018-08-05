

#==================================================
#  Class that defines different operations related
#  to final content extraction
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013 - 2015
#
#   - Kenny Davila (June, 2015)
#      - refactoring
#==================================================

import math

import cv
import cv2

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.preprocessing.data.change_region import ChangeRegion


class Extractor:
    #========================================================
    #  Gets the RGB images corresponding to a change region
    #  (Region of interest) from the main videos
    #========================================================
    @staticmethod
    def extractRegions( main_videos, speaker_detector, change_detector, alignment, time_offset):
        #sort the input regions
        #by last time modified...
        regions = change_detector.getAllRegions()
        sorted_regions = [ (r.last_modified, idx, r) for idx, r in enumerate(regions) ]
        sorted_regions = sorted( sorted_regions )

        open_capture = None
        open_index = None

        extracted_parts = {}
        extracted_boxes = {}

        w_cells = change_detector.getWidthCells()
        h_cells = change_detector.getHeightCells()

        motions = speaker_detector.getMotionDetected()

        #now extract them from the videos
        for time_modified, idx, r in sorted_regions:
            #for the region r, check one frame between
            #last time modified and time locked where
            #it is assumed to be not obstructed...
            real_time_modified = time_modified + time_offset

            init_index =  speaker_detector.findMotionByTime( real_time_modified )
            if init_index == len(motions):
                init_index -= 1

            candidates = []

            #find the corresponding region of interest in original video...
            #...first... from cells to rectangle in original frame....
            cx_min = (r.min_x / float(w_cells)) * alignment.aux_width
            cx_max = ((r.max_x + 1) / float(w_cells)) * alignment.aux_width
            cy_min = (r.min_y / float(h_cells)) * alignment.aux_height
            cy_max = ((r.max_y + 1) / float(h_cells)) * alignment.aux_height
            #...then... use scaling information to obtain corresponding
            #... rectangle in main video....
            main_region = alignment.alignRegion( cx_min, cx_max, cy_min, cy_max )

            #<THE MASK>
            """
            #generate the mask
            mask = r.getMask(h_cells, w_cells)

            #resize...
            new_size = (int(mask.shape[1] * ChangeDetector.CELL_SIZE), int(mask.shape[0] * ChangeDetector.CELL_SIZE) )
            mask = cv2.resize(mask, new_size)

            #project the mask
            proj_mask = np.zeros( (alignment.main_height, alignment.main_width) , dtype='uint8' )
            cv.WarpPerspective( cv.fromarray( mask ), cv.fromarray(proj_mask), cv.fromarray( alignment.projection ) )

            #dilate the mask
            #....create structuring element...
            expand_cells = 4
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ChangeDetector.CELL_SIZE * expand_cells), int(ChangeDetector.CELL_SIZE * expand_cells)))

            #....now dilate mask...
            final_mask = cv2.dilate(proj_mask, strel)
            """
            #</THE MASK>

            #now, find a frame where the found region has no motion around
            intervals = speaker_detector.getNonblockedIntervals(main_region, alignment.main_width,
                                                                alignment.main_height, init_index,
                                                                r.locked_time + time_offset)

            #check....
            if len(intervals) == 0:
                #cannot be extracted on lock time...
                #now do a second attempt to extract it based on erasing time...
                if r.lock_type == ChangeRegion.COMPLETE_ON_OVERWRITE and \
                   r.erased_time != None:
                    intervals = speaker_detector.getNonblockedIntervals(main_region, alignment.main_width,
                                                                        alignment.main_height, init_index,
                                                                        r.erased_time + time_offset)
            #check....
            if len(intervals) == 0:
                #it simply could not be extracted....
                extracted_parts[ r.number_id ] = None
                extracted_boxes[ r.number_id ] = None
                print( "CHECK = "  + str(r.number_id ) + ", from " + \
                       TimeHelper.stampToStr(r.creation_time + time_offset) + " - " + \
                       TimeHelper.stampToStr(motions[init_index].absolute_time) + " to " + \
                       TimeHelper.stampToStr(r.locked_time + time_offset) )
            else:

                #find the best interval....
                best_interval = 0
                best_length = (intervals[0][1] - intervals[0][0] + 1)
                for i in range(1, len(intervals)):
                    length = (intervals[i][1] - intervals[i][0] + 1)
                    if length > best_length:
                        best_length = length
                        best_interval = i

                #now, from best interval pick frame in the middle..
                best_frame = int(init_index + ((intervals[best_interval][0] + intervals[best_interval][1]) / 2.0))

                #finally do extraction
                #... open video ...
                if open_index == None or open_index != motions[best_frame].video_index:
                    #close current video
                    if open_index != None:
                        open_capture.release()

                    open_index = motions[best_frame].video_index
                    open_capture = cv2.VideoCapture(main_videos[open_index] )

                #... set time position ...
                open_capture.set(cv.CV_CAP_PROP_POS_MSEC, motions[best_frame].time)
                #... get frame ...
                flag, frame = open_capture.read()

                if not flag:
                    #error?
                    extracted_parts[ r.number_id ] = None
                    extracted_boxes[ r.number_id ] = None
                    print( "Region <" + str(r.number_id) + "> Could not be extracted from video" )
                else:
                    #extract the region ...
                    margin = 5
                    min_x = int(max(0, math.floor(main_region[0] - margin) ))
                    max_x = int(min(alignment.main_width - 1, math.ceil(main_region[1] + margin) ))
                    min_y = int(max(0, math.floor(main_region[2] - margin) ))
                    max_y = int(min(alignment.main_height - 1, math.ceil(main_region[3] + margin) ))

                    #get the part of the image...
                    part = frame[  min_y:max_y, min_x:max_x, :]
                    #<THE MASK>
                    """
                    part_mask = final_mask[min_y:max_y, min_x:max_x]

                    size_mask = int((part_mask.shape[0] * part_mask.shape[1]) - np.count_nonzero(part_mask))

                    if size_mask > 0:
                        original_part = part.copy()

                        blured_part = cv2.GaussianBlur(original_part, (41, 41), 4.0)

                        part[part_mask == 0, :] = blured_part[part_mask == 0, :]

                    """
                    #</THE MASK>

                    extracted_parts[ r.number_id ] = part
                    extracted_boxes[ r.number_id ] = (min_x, max_x, min_y, max_y)

                    print( "Region <" + str(r.number_id) + "> extracted succesfully!" )

        return extracted_parts, extracted_boxes