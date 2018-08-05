
#==================================================
#  Class that defines operations related to
#  loading content from videos
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     June 2015
#
#==================================================

import cv2

class Loader:
    #==================================================================
    #   extract video frames given the video index and time location
    #   video_files = list of file names of videos
    #   frame_list = list of:
    #           (index, time) index of the video and relative time
    #   returns list of (absolute time, frame)
    #==================================================================
    @staticmethod
    def extractFramesRelative(video_files, frame_list):
        frames = []

        frame_dict = {}
        #put frames on separated list if they come from separated videos
        for video_index, frame_time in frame_list:
            if video_index in frame_dict:
                frame_dict[video_index].append( frame_time )
            else:
                frame_dict[video_index] = [ frame_time ]

        absolute_time = 0.0
        for idx, video_file in enumerate(video_files):
            #open video
            try:
                capture = cv2.VideoCapture(video_file)
            except:
                #error loading
                return None

            if not capture.isOpened():
                #video could not be opened
                return None

            if idx in frame_dict:
                for frame_time in frame_dict[idx]:
                    #set time for extraction...
                    capture.set(cv2.CAP_PROP_POS_MSEC, frame_time)

                    #get...
                    flag, frame = capture.read()

                    if flag:
                        frames.append( (absolute_time + frame_time, frame ) )

            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            video_length = capture.get(cv2.CAP_PROP_POS_MSEC)

            absolute_time += video_length

        return frames

    #==================================================================
    #   extract video frames given their absolute times
    #   video_files = list of file names of videos
    #   frame_list = list of:
    #           (absolute times) absolute time of the frame
    #   returns list of (frame)
    #==================================================================
    @staticmethod
    def extractFramesAbsolute(video_files, frame_list):
        frames = []

        #ensure that times are sorted...
        frame_list = sorted( frame_list )

        absolute_time = 0.0
        pos = 0
        for video_file in video_files:
            #open video
            try:
                capture = cv2.VideoCapture(video_file)
            except:
                #error loading
                return None

            if not capture.isOpened():
                #video could not be opened
                return None

            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            video_length = capture.get(cv2.CAP_PROP_POS_MSEC)

            video_start = absolute_time
            video_end = absolute_time + video_length

            while pos < len(frame_list):
                if frame_list[pos] >= video_end:
                    #the frame belongs to next segment...
                    break
                else:
                    #belongs to current video...
                    #....go to desired position in time...
                    capture.set(cv2.CAP_PROP_POS_MSEC, frame_list[pos] - video_start)

                    #get frame...
                    flag, frame = capture.read()

                    if flag:
                        frames.append( frame )

                    pos += 1

            #add the length of current segment....
            absolute_time += video_length

        return frames
