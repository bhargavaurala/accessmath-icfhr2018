
#==========================================================
# CLASS THAT REPRESENTS A REGION OF CONTENT THAT LATER
# WILL BECOME A SKETCH
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#
# Modified by:
#    - Kenny Davila (May 5, 2014)
#      - Added "erased_time" to string representation
#==========================================================

import cv2
import numpy as np

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.preprocessing.data.change_cell import ChangeCell


class ChangeRegion:
    CLOSE_ON_TOP = 2
    CLOSE_ON_BOTTOM = 3
    CLOSE_ON_LEFT = 3
    CLOSE_ON_RIGHT = 10

    COMPLETE_UNDEFINED = 0
    COMPLETE_ON_OVERWRITE = 1
    COMPLETE_ON_ERASING = 2
    COMPLETE_ON_EOV = 3

    def __init__(self, number_id, time, max_width, max_height):
        #id...
        self.number_id = number_id
        #time of creation...
        self.creation_time = time
        #cells that compose the region
        self.cells = []
        #last time modified...
        self.last_modified = -1.0
        #boundaries of the region....
        self.min_x = max_width
        self.max_x = 0
        self.min_y = max_height
        self.max_y = 0
        #end of life span...
        self.locked_time = -1.0 #Unknown
        #real end of time on board
        self.erased_time = None #Unknown

        #for accepted deletions...
        self.boundaries_outdated = False

        #for overwritten case
        self.overwritten_by = None

        #not yet locked...
        self.lock_type = ChangeRegion.COMPLETE_UNDEFINED

    def getEditionTime(self):
        return self.last_modified - self.creation_time

    def getDensity(self):
        return len(self.cells) / float(self.getWidth() * self.getHeight())

    def getCount(self):
        return len(self.cells)

    def getWidth(self):
        return self.max_x - self.min_x + 1

    def getHeight(self):
        return self.max_y - self.min_y + 1

    def getArea(self):
        return self.getWidth() * self.getHeight()

    def getAspectRatio(self):
        return float(self.getWidth()) / float(self.getHeight())

    def recentlyModified(self, current_time):
        #in milliseconds
        time_window = 10000.0

        return self.last_modified + time_window >= current_time

    def closeEnough(self, x, y ):
        if self.boundaries_outdated:
            self.updateBoundaries()

        return len(self.cells) > 0 and \
               x <= self.max_x + ChangeRegion.CLOSE_ON_RIGHT and \
               x >= self.min_x - ChangeRegion.CLOSE_ON_LEFT and \
               y <= self.max_y + ChangeRegion.CLOSE_ON_BOTTOM and \
               y >= self.min_y - ChangeRegion.CLOSE_ON_TOP

    def scaledCloseEnough(self, x, y, x_scale, y_scale ):
        if self.boundaries_outdated:
            self.updateBoundaries()

        return len(self.cells) > 0 and \
               x <= self.max_x + ChangeRegion.CLOSE_ON_RIGHT * x_scale and \
               x >= self.min_x - ChangeRegion.CLOSE_ON_LEFT * x_scale and \
               y <= self.max_y + ChangeRegion.CLOSE_ON_BOTTOM * y_scale and \
               y >= self.min_y - ChangeRegion.CLOSE_ON_TOP * y_scale

    def addCell(self, cell_reference, time ):
        assert isinstance(cell_reference, ChangeCell)

        #try making change in cell first...
        cell_reference.addToRegion( self )

        #add to list...
        self.cells.append( cell_reference )

        #check if new boundaries...
        if cell_reference.pos_x < self.min_x:
            self.min_x = cell_reference.pos_x
        if self.max_x < cell_reference.pos_x:
            self.max_x = cell_reference.pos_x
        if cell_reference.pos_y < self.min_y:
            self.min_y = cell_reference.pos_y
        if self.max_y < cell_reference.pos_y:
            self.max_y = cell_reference.pos_y

        #last time modified...
        self.last_modified = time


    def updatedCell( self, cell_reference, time ):
        assert isinstance(cell_reference, ChangeCell)

        if cell_reference.current_region.number_id == self.number_id:
            self.last_time_modified = time

    def lockRegion( self, time, lock_type, overwriter_id):
        #check if not already locked....
        if self.isLocked():
            return

        #store the max time on which this region can be
        #extracted from the whiteboard video...
        self.locked_time = time

        #check the type of lock
        self.lock_type = lock_type
        if lock_type != ChangeRegion.COMPLETE_ON_OVERWRITE:
            #the time of erasing is known
            self.erased_time = time

        #for overwritten case
        self.overwritten_by = overwriter_id

        #check if boundaries outdated...
        if self.boundaries_outdated:
            self.updateBoundaries()

        #now, release all cells...
        for cell in self.cells:
            #mark as a free cell again...
            cell.releaseFromRegion(lock_type != ChangeRegion.COMPLETE_ON_OVERWRITE)

    def isLocked( self):
        return self.locked_time >= 0.0

    def __str__( self) :
        content = "ChangeRegion -> ID: " + str(self.number_id ) + "\n"
        content += " Time: " + TimeHelper.stampToStr( self.creation_time )
        content += " - " + TimeHelper.stampToStr( self.last_modified )
        if self.locked_time >= 0.0:
            content += " - " + TimeHelper.stampToStr( self.locked_time )

        if self.erased_time is None:
            content += "\n"
        else:
            content += " - " + TimeHelper.stampToStr( self.erased_time )+ "\n"
        content += " Region: X: [" + str(self.min_x) + ", " + str(self.max_x) + "]"
        content += ", Y: [" + str(self.min_y) + ", " + str(self.max_y) + "]\n"
        content += " Cells: " + str(len(self.cells))

        return content

    def mergeRegions( self, other_region, time ):
        #update references in cell
        for cell in other_region.cells:
            #... tell cell that it belongs to a new region
            cell.addToMergedRegion( self )

            #.... also, add cell into local region array
            self.cells.append( cell )

        #adjust region boundaries
        if other_region.min_x < self.min_x:
            self.min_x = other_region.min_x
        if self.max_x < other_region.max_x:
            self.max_x = other_region.max_x
        if other_region.min_y < self.min_y:
            self.min_y = other_region.min_y
        if self.max_y < other_region.max_y:
            self.max_y = other_region.max_y

        #in case any of the two had deleted elements
        self.boundaries_outdated = self.boundaries_outdated or other_region.boundaries_outdated

        #update last time modified
        self.last_modified = time

    def removeCell( self, cell ):
        #remove from the array of cells...
        self.cells.remove( cell )
        #boundaries need to be updated
        self.boundaries_outdated = True

    def isEmpty(self):
        return len(self.cells) == 0

    def updateBoundaries(self):
        if len(self.cells) == 0:
            #can't update boundaries of an empty region
            return

        #restart boundaries...
        self.min_x = self.cells[0].pos_x
        self.max_x = self.cells[0].pos_x
        self.min_y = self.cells[0].pos_y
        self.max_y = self.cells[0].pos_y

        #found new boundaries...
        for cell in self.cells:
            #check if new boundaries...
            if cell.pos_x < self.min_x:
                self.min_x = cell.pos_x
            if self.max_x < cell.pos_x:
                self.max_x = cell.pos_x
            if cell.pos_y < self.min_y:
                self.min_y = cell.pos_y
            if self.max_y < cell.pos_y:
                self.max_y = cell.pos_y

        self.boundaries_outdated = False

    def saveAsImage(self, file_name, h_cells, w_cells):
        mat = np.zeros( (h_cells, w_cells, 3) )

        #mark cells
        for cell in self.cells:
            mat[ cell.pos_y, cell.pos_x, 1] = 255


        cv2.imwrite(file_name, mat)

    def getMask(self, h_cells, w_cells):
        if self.boundaries_outdated:
            self.updateBoundaries()

        mat = np.zeros( (h_cells, w_cells) , dtype='uint8' )

        #mark cells...
        for cell in self.cells:
            mat[ cell.pos_y, cell.pos_x] = 255

        return mat