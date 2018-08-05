
#======================================================
# Auxiliary class to store information about visual
# alignment of two videos from the white board
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#
# Modified by:
#    - Kenny Davila (May 5, 2014)
#      - Avoid doing RANSAC if less than 4 matching points are provided...
#
#======================================================


import numpy as np
import cv2

class VisualAlignment:
    def __init__(self):
        #start with default values...

        #for main video resolution
        self.main_width = 100
        self.main_height = 75
        #...and whiteboard box ...
        self.main_box = (0,100,0,75)

        #for auxiliary video resolution
        self.aux_width = 100
        self.aux_height = 75
        #...and whiteboard box ...
        self.aux_box = (0,100,0,75)

        #alignment as projection matrix...
        #but initially undefined...
        self.projection = None

    def __str__(self):
        content  = "Visual Alignment\n"
        content += " -> Main Size ( " + str(self.main_width) + ", " + str(self.main_height) + ") \n"
        content += " -> Second Size (" + str(self.aux_width) + ", " + str(self.aux_height) + ") \n"

        content += " -> Main Box ( [" + str(self.main_box[0]) + ", " + str(self.main_box[1])
        content += "], [" + str(self.main_box[2]) + ", " + str(self.main_box[3]) + "] ) \n"

        content += " -> Second Box ( [" + str(self.aux_box[0]) + ", " + str(self.aux_box[1])
        content += "], [" + str(self.aux_box[2]) + ", " + str(self.aux_box[3]) + "] ) \n"

        if self.projection != None:
            content += " -> Projection: " + str(self.projection) + "\n"

        return content

    #=======================================================
    # Find the bounding box that contains the projection of
    # another box
    #=======================================================
    def alignRegion(self, min_x, max_x, min_y, max_y):

        #project the corners of the box....
        corners = np.array( [ [min_x, min_y],
                              [min_x, max_y],
                              [max_x, min_y],
                              [max_x, max_y] ], dtype='float32')

        corners = np.array([corners])
        projected = cv2.perspectiveTransform(corners, self.projection)

        new_min_x = projected[0,:,0].min()
        new_max_x = projected[0,:,0].max()
        new_min_y = projected[0,:,1].min()
        new_max_y = projected[0,:,1].max()

        return (new_min_x, new_max_x, new_min_y, new_max_y)

    #======================================================================
    # Gets two list of points that are believed to be equivalent between
    # the two given images. The system generates key points and
    # describes them using Speed Up Robust Features (SURF) and then
    # finds good matches with distance below the given threshold.
    #
    # Based on the code shown on this forum:
    #     http://stackoverflow.com/questions/10984313/opencv-2-4-1-computing-surf-descriptors-in-python
    #======================================================================
    @staticmethod
    def getSURFMatchingPoints(img_object_gray, img_scene_gray, threshold):
        # SURF extraction
        surf = cv2.SURF(400, extended = True)
        #surf = cv2.FeatureDetector_create("SURF") #SURF
        surfDescriptor = cv2.DescriptorExtractor_create("SURF")

        #...main scene...
        key_points_scene  = surf.detect( img_scene_gray )
        (key_points_scene, scene_descriptors) = surfDescriptor.compute( img_scene_gray, key_points_scene)
        #..."object"....
        key_points_object  = surf.detect( img_object_gray )
        (key_points_object, object_descriptors) = surfDescriptor.compute( img_object_gray, key_points_object)

        #check if key points in main scene...
        if len(key_points_scene) > 0:
            # Setting up samples and responses for kNN
            samples = np.array(scene_descriptors)
            responses = np.arange(len(key_points_scene),dtype = np.float32)

            # kNN training
            knn = cv2.KNearest()
            knn.train(samples, responses)

        #USE kNN for matching...
        object_points = []
        scene_points = []

        #do matching iff there are key points in object...
        if len(key_points_object) > 0:
            for idx_object, descriptor in enumerate(object_descriptors):
                descriptor = np.array(descriptor,np.float32).reshape((1,descriptor.shape[0]))
                retval, results, neigh_resp, dists = knn.find_nearest(descriptor,1)
                idx_scene, distance =  int(results[0][0]), dists[0][0]

                if distance < threshold:
                    object_points.append(key_points_object[ idx_object ].pt)
                    scene_points.append(key_points_scene[ idx_scene ].pt)

        return object_points, scene_points

    #===============================================================
    # Takes two list of corresponding points between two 2D-planes
    # and finds a projection matrix that produce an acceptable
    # mapping between the two planes
    #===============================================================
    @staticmethod
    def generateProjection(object_list, scene_list ):
        #cannot match less than 4 points
        if len(object_list) < 4:
            return None, None

        # use the matches to extract points
        object_points = np.zeros( (len(object_list), 2) )
        scene_points = np.zeros( (len(object_list), 2) )
        for idx, object_point in enumerate(object_list):
            object_points[idx, :] = object_point
            scene_points[idx, :] = scene_list[idx]

        #find the projection
        projection, mask = cv2.findHomography( object_points, scene_points, cv2.RANSAC, ransacReprojThreshold=3 )
        #projection, mask = cv2.findHomography( object_points, scene_points, cv2.LMEDS )

        return projection, mask

    #==============================================================
    # Gets two images of the same element and finds a projection
    # matrix for image registration
    #==============================================================
    @staticmethod
    def getProjection(img_object_gray, img_scene_gray, threshold):

        object_list, scene_list = VisualAlignment.getSURFMatchingPoints(img_object_gray, img_scene_gray, threshold)

        projection, mask = VisualAlignment.generateProjection( object_list, scene_list )

        return projection

    #==============================================================
    # Assigns the input project a score based on the recall of
    # dark pixels of the original content matched by the projection
    #==============================================================
    @staticmethod
    def getProjectionScore( projection, scene_list, object_list ):
        if projection is None:
            #lowest possible score for undefined projections
            return 0.0

        avg_recall = 0.0
        for i in range(len(scene_list)):
            content_scene = scene_list[i]
            content_object = object_list[i]

            copy_object = content_object.copy()
            copy_object[ copy_object == 0 ] = 128

            proj_img = np.zeros( (content_scene.shape[0], content_scene.shape[1]), dtype=content_scene.dtype )
            cv.WarpPerspective( cv.fromarray( copy_object ), cv.fromarray(proj_img), cv.fromarray( projection ) )

            proj_img[ proj_img != 128 ] = 255
            proj_img[ proj_img == 128 ] = 0

            combined = cv2.bitwise_or(content_scene, proj_img)

            total_pixels = float(combined.shape[0] * combined.shape[1])

            content_pixels = total_pixels - np.count_nonzero(content_scene)
            combined_background = np.count_nonzero(combined)
            total_matched =  total_pixels - combined_background

            recall = total_matched / float(content_pixels)

            avg_recall += recall

        avg_recall /= float( len(scene_list) )

        return avg_recall * 100.0

