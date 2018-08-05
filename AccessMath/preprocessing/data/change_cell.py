
#=============================================
# CLASS THAT REPRESENTS A CELL OF THE GRID
# USED BY THE CHANGE DETECTION ALGORITHM
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#=============================================

class ChangeCell:
    TONE_THRESHOLD = 204
    VALUE_CHANGE = 30

    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

        self.modified_frame = -1
        self.modified_time = -1.0

        self.extracted_frame = -1
        self.extracted_time = -1.0

        self.erased_frame = -1
        self.erased_time = -1.0

        self.tone = 255.0 #assume white....

        self.written = False
        self.erased = False

        self.current_region = None
        self.last_region = None

    def isDirty(self):
        return self.modified_frame > self.extracted_time

    def changeTone(self, frame, time, new_tone):

        if new_tone > self.tone:
            if self.written and ChangeCell.TONE_THRESHOLD <= new_tone:
                self.erased = True
                self.written = False
                self.erased_frame = frame
                self.erased_time = time
        else:
            self.erased = False
            if new_tone < ChangeCell.TONE_THRESHOLD <= self.tone:
                self.written = True

        self.tone = new_tone
        self.modified_frame = frame
        self.modified_time = time

    def addToRegion( self, region):
        #added to that region
        self.current_region = region

    def addToMergedRegion(self, new_region ):
        #update references ...
        #replace current region
        self.current_region = new_region

    def releaseFromRegion(self, complete_on_erasing):
        if complete_on_erasing:
            self.last_region = None
        else:
            self.last_region = self.current_region

        self.current_region = None
