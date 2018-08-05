#=====================================================================
#  Class that defines different operations related
#  to a region of content (sketch)
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013 - 2015
#
#   Modified by:
#       - Kenny Davila (June 9, 2014)
#         - Added extraction of image with color highlighting
#       - Kenny Davila (January 19, 2015)
#         - Added cached pre-processing
#         - Added C feature extraction
#       - Kenny Davila (June 22, 2015)
#         - Renamed as ContentRegion (previously called Sketch)
#=====================================================================

import cv2

class ContentRegion:
    UseCachedPreprocessing = True
    CacheLocation = 'cache'

    def __init__(self, s_id, creation_time, last_modified, time_locked, lock_type, ow_id, time_erased, box, image,
                 name=None):
        self.id = s_id
        self.creation_time = creation_time
        self.last_modified = last_modified
        self.time_locked = time_locked
        self.lock_type = lock_type
        self.overwritten_by = ow_id
        self.time_erased = time_erased
        self.box = box
        self.image = image
        self.name = name

        self.content = None
        self.components = None

    def saveImage(self, path):
        cv2.imwrite(path, self.image)