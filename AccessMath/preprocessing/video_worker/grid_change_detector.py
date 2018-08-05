
#===================================================================
# Class that holds the routines required for detection of changes
# in the content of the white board in a video based on a grid
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#
# Modified by:
#    - Kenny Davila (May 5, 2014)
#       - now handles case where video starts black for a few seconds...
#===================================================================

import math

import cv
import cv2
import numpy as np

from AccessMath.preprocessing.data.change_cell import ChangeCell
from AccessMath.preprocessing.data.change_region import ChangeRegion


class GridChangeDetector:
    CELL_SIZE = 4.0
    VALUE_CHANGE = 30

    def __init__(self):
        #prepare...
        self.width = None
        self.height = None

        #grid resolution
        self.w_cells = 0
        self.h_cells = 0

        self.cells = None

        self.all_regions = None
        self.current_regions = None
        self.next_region_id = None

        self.last_modified = None
        self.last_time = 0.0
        self.frame_count = 0
        self.last_bw = None
        self.last_grayscale = None

        self.check_silence = True

    def initialize(self, width, height):
        #video resolution...
        self.width = width
        self.height = height

        #grid resolution
        self.w_cells = int(math.ceil(width / GridChangeDetector.CELL_SIZE))
        self.h_cells = int(math.ceil(height / GridChangeDetector.CELL_SIZE))

        self.cells = [[ ChangeCell(x, y) for x in range(self.w_cells) ] for y in range(self.h_cells) ]

        self.all_regions = []
        self.current_regions = []
        self.next_region_id = 1

        self.last_modified = [[ -1.0 for c in range(self.w_cells)] for r in range(self.h_cells) ]
        self.last_time = 0.0
        self.frame_count = 0
        self.last_bw = None
        self.last_grayscale = None

        self.check_silence = True


    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time):
        #check if completely black at beginning...
        if self.check_silence:
            previous_mean = last_frame.mean()
            if previous_mean < 10.0:
                #still silence... do nothing...
                return
            else:
                #will not skip any more initial frames...
                self.check_silence = False

        #first, convert to int32 to do substraction...
        tone_threshold = 204
        grayscale = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
        current_bw = cv2.compare(grayscale, tone_threshold, cv2.CMP_GT)

        if self.last_bw is None:
            self.last_grayscale = cv2.cvtColor(last_frame, cv.CV_RGB2GRAY)
            self.last_bw = cv2.compare(self.last_grayscale, tone_threshold, cv2.CMP_GT)

        #compare for changes...
        diff = cv2.compare( current_bw, self.last_bw, cv2.CMP_NE )
        dist = cv2.absdiff( grayscale, self.last_grayscale)

        #get the positions changed....
        y, x = np.nonzero( np.bitwise_and( cv2.compare(dist, GridChangeDetector.VALUE_CHANGE, cv2.CMP_GT), diff ) )

        changes_detected = []
        for i in xrange(len(y)):
            cell_x = int( x[i] / GridChangeDetector.CELL_SIZE)
            cell_y = int( y[i] / GridChangeDetector.CELL_SIZE)

            if self.last_modified[cell_y][cell_x] < abs_time:
                changes_detected.append( (cell_x, cell_y) )
                self.last_modified[cell_y][cell_x] = abs_time

        #for each change detected ....
        cells_written = []
        cells_erased = []
        for x, y in changes_detected:
            size = int(GridChangeDetector.CELL_SIZE)
            min_x = size * x
            min_y = size * y

            #use the average of the tones... or
            #tone = np.average(grayscale[min_y:(min_y + size), min_x:(min_x + size)] )
            #use the darkest pixel on the cell as reference
            tone = np.min(grayscale[min_y:(min_y + size), min_x:(min_x + size)] )

            #update cell...
            self.cells[y][x].changeTone( self.frame_count, abs_time, tone )



            #add to list if modified
            if self.cells[y][x].written:
                #was written
                cells_written.append( (x, y) )
            elif self.cells[y][x].erased:
                #was erased
                cells_erased.append( (x, y) )

        #process first all the erased cells!
        for x, y in cells_erased:
            region = self.cells[y][x].current_region

            #for cells still associated to regions...
            if region != None:
                #is region active?
                if region.recentlyModified(abs_time):
                    #it is still in "edit" mode...
                    #just remove cell from current region
                    region.removeCell( self.cells[y][x] )
                    self.cells[y][x].current_region = None
                    self.cells[y][x].last_region = None

                    if region.isEmpty():
                        #remove from current regions...
                        self.current_regions.remove( region )
                else:
                    #erasing a piece of a completed region
                    region.lockRegion( self.last_time, ChangeRegion.COMPLETE_ON_ERASING, None )

            else:
                last_region = self.cells[y][x].last_region
                #if the cell was associated to a previous region...
                if last_region != None:
                    #it was part of a commited (locked) region
                    #check if that region was mark as erased ...
                    if last_region.erased_time == None:
                        #it was unknown, and now it is known...
                        last_region.erased_time = self.last_time

                    #remove the link from that region
                    self.cells[y][x].last_last_region = None


        for x, y in cells_written:
            #check....
            if len(self.current_regions) == 0:
                #no active regions, add new one...
                new_region = ChangeRegion( self.next_region_id, abs_time, self.w_cells, self.h_cells )
                self.next_region_id += 1
                self.current_regions.append( new_region )

                #add cell to region...
                new_region.addCell(self.cells[y][x], abs_time )
            else:
                #active regions exist...
                if self.cells[y][x].current_region == None:
                    #its a free cell...
                    #check for all active regions ....
                    close_regions = []
                    for r in self.current_regions:

                        if r.closeEnough( x, y ) and (not r.isLocked()):
                            #check....
                            if r.recentlyModified(abs_time):
                                #is a valid active region...
                                valid = True

                                #don't join with lines (if not part of the line)....
                                if r.getAspectRatio() < 0.1 and \
                                   r.getHeight() / float(self.h_cells) > 0.4 and \
                                   not (r.scaledCloseEnough(x, y, 0.5, 1.0 )):
                                    valid = False

                                if valid:
                                    close_regions.append( r )
                            else:
                                #is an old region, mark as locked
                                #before creating an overlapping region
                                r.lockRegion( self.last_time, ChangeRegion.COMPLETE_ON_OVERWRITE, self.next_region_id )


                    if len(close_regions) > 0:
                        #Close to active regions

                        #check for undesired merges...
                        if len(close_regions) > 1:
                            rejected = True
                            total_cells = self.w_cells * self.h_cells
                            while rejected and len(close_regions) > 1:
                                rejected = False
                                #calculate the size of the resulting region....
                                min_x = close_regions[0].min_x
                                min_y = close_regions[0].min_y
                                max_x = close_regions[0].max_x
                                max_y = close_regions[0].max_y
                                larger = 0
                                for i in range(1, len(close_regions)):
                                    min_x = min( min_x, close_regions[i].min_x )
                                    min_y = min( min_y, close_regions[i].min_y )
                                    max_x = max( max_x, close_regions[i].max_x )
                                    max_y = max( max_y, close_regions[i].max_y )

                                    if close_regions[larger].getArea() < close_regions[i].getArea():
                                        larger = i

                                merged_size = (max_x - min_x + 1) * (max_y - min_y + 1)
                                large_size = close_regions[larger].getArea()


                                if large_size > total_cells * 0.09 and \
                                   merged_size - large_size > total_cells * 0.01 and \
                                   min_y < close_regions[larger].min_y:
                                    #don't merge when texts seems to belong to a new column

                                    #The region is large, the addition will make it even larger
                                    #and one of the regions to add is over the largest region
                                    #(probably a different text column)
                                    #therefore, exclude larger region to avoid undesired merge
                                    #of two columns of writing on the white board
                                    del close_regions[larger]
                                    rejected = True

                                elif close_regions[larger].getAspectRatio() < 0.10 and \
                                    close_regions[larger].getHeight() / float(self.h_cells) > 0.4 :
                                    #don't merge if larger seems to be a vertical line (division) on the board

                                    #The region has the aspect ratio of a vertical line with
                                    #that covers a large part of the whiteboard, likely to be a division line
                                    del close_regions[larger]
                                    rejected = True


                        #choose the smallest id....
                        chosen = 0
                        for i in range(1, len(close_regions)):
                            if close_regions[i].number_id < close_regions[chosen].number_id:
                                chosen = i

                        #add to that region...
                        close_regions[chosen].addCell(self.cells[y][x], abs_time )

                        #now merge regions (if multiple close regions)....
                        for i in range(len(close_regions)):
                            if i != chosen:
                                #....add cells from each other regions to the chosen ...
                                close_regions[chosen].mergeRegions( close_regions[i], abs_time )

                                #remove region from current_regions
                                k = 0
                                #...search for it....
                                while k < len(self.current_regions):
                                    if self.current_regions[k] == close_regions[i]:
                                        #found... remove....
                                        del self.current_regions[k]
                                        #stop searching...
                                        break
                                    else:
                                        k += 1
                    else:
                        #not close to active region
                        #... Create a new region
                        new_region = ChangeRegion( self.next_region_id, abs_time, self.w_cells, self.h_cells )
                        self.next_region_id += 1
                        self.current_regions.append( new_region )

                        #... add cell to region...
                        new_region.addCell(self.cells[y][x], abs_time )
                else:
                    #it is not a free cell.. check...
                    region = self.cells[y][x].current_region
                    if region.recentlyModified(abs_time):
                        #Belongs to active region...
                        #Just update region....
                        region.updatedCell( self.cells[y][x], abs_time )
                    else:
                        #the region is old, mark as locked...
                        region.lockRegion( self.last_time, ChangeRegion.COMPLETE_ON_OVERWRITE, self.next_region_id )

                        #then, overwrite with a new region
                        new_region = ChangeRegion( self.next_region_id, abs_time, self.w_cells, self.h_cells )
                        self.next_region_id += 1
                        self.current_regions.append( new_region )

                        #add cell to region...
                        new_region.addCell(self.cells[y][x], abs_time )

        """
        if self.next_region_id > 1000:
            for idx, region in enumerate(self.all_regions):
                region.saveAsImage("out/region_" + str(region.number_id) + ".jpg", self.h_cells, self.w_cells)
        """

        #check for regions marked as locked....
        pos = 0
        while pos < len(self.current_regions):
            #if it is locked....
            if self.current_regions[pos].isLocked():
                #....add to history of regions....
                self.all_regions.append( self.current_regions[pos] )
                #....remove from current_regions...
                del self.current_regions[pos]
            else:
                #check if boundaries outdated...
                if self.current_regions[pos].boundaries_outdated:
                    self.current_regions[pos].updateBoundaries()
                #go to next region....
                pos += 1

        self.last_time = abs_time
        self.frame_count += 1
        #save... so gray scale conversion is done only once per frame
        self.last_grayscale = grayscale
        self.last_bw = current_bw

    def getWorkName(self):
        return "Change Detection"

    def finalize(self):
        #at this point, if active regions remain, mark them as locked
        #and add them to history of region....
        for region in self.current_regions:
            region.updateBoundaries()
            #lock ....
            region.lockRegion( self.last_time, ChangeRegion.COMPLETE_ON_EOV, None )
            #add to history....
            self.all_regions.append( region )

    def getAllRegions(self):
        return self.all_regions

    def getWidthCells(self):
        return self.w_cells

    def getHeightCells(self):
        return self.h_cells

    def filterRegions(self):
        #sort regions by region_id
        tempo_regions =  sorted([ (r.number_id, r) for r in self.all_regions ])
        self.all_regions =  [ r for idx, r in tempo_regions ]

        #FILTER REGIONS
        pos = 0
        while pos < len(self.all_regions):
            region = self.all_regions[pos]

            #use four rules to remove regions that are likely to be noise
            # 1) If the number of cells is too small to represent
            #    relevant content
            # 2) if the region was edite within a time span too short to
            #    belong to relevant content
            # 3) if the density of the content (content/area) is too small
            #    to represent a real drawing
            # 4) if Aspect ratio makes the region likely to be a line
            if region.getCount() < 20 or \
               region.getEditionTime() < 400 or \
               region.getDensity() < 0.1 or \
               region.getAspectRatio() < (1 / 10.0) or \
               region.getAspectRatio() > 10.0:
                toFilter = True

            #exceptions to the rules....
            #...low density/edition time, but too big to be noise....
            if region.getAspectRatio() > (1 / 10.0) and \
               region.getAspectRatio() < 10.0 and \
               region.getCount() >= 35:
                #regardless edition time or density, if it has more than 50 cells,
                #and the right aspect ratio, then do not filter
                toFilter = False

            #...low size, acceptable on everything else...
            if region.getAspectRatio() > (1 / 10.0) and \
               region.getAspectRatio() < 10.0 and \
               region.getDensity() >= 0.1 and \
               region.getEditionTime() >= 400 and \
               region.getCount() >= 10:
                toFilter = False

            #...very wide fails the aspect ratio test for lines,
            #...but edition time is too large to be a single line
            if region.getAspectRatio() > 1.0 and \
               region.getAspectRatio() <= 15.0 and \
               region.getDensity() >= 0.1 and \
               region.getEditionTime() >= 5000 and \
               region.getCount() >= 10:
                toFilter = False

            if toFilter:
                #remove noisy region
                del self.all_regions[pos]
            else:
                pos += 1

