import cv2

from AM_CommonTools.util.time_helper import TimeHelper


# ===================================================================
# Abstraction for Video Processing. Since many algorithms require
# to apply operations using frame differencing, this class handles
# the skeleton for such operations, leaving the specific operations
# to be handled by an external classes
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#
# Modified:
#     October 2015
#         - By Kenny Davila: Adaptive sampling added.
# ===================================================================

class VideoProcessor:
    
    def __init__(self, file_list, frames_per_second=1):
        self.file_list = file_list
        self.frames_per_second = frames_per_second
        self.forced_width = None
        self.forced_height = None

    def force_resolution(self, width, height):
        self.forced_width = width
        self.forced_height = height

    def checkError(self):
        print("Works?")

    def doProcessing(self, video_worker, limit=0, verbose=False):
        #initially....
        width = None
        height = None

        offset_frame = -1
        absolute_frame = 0
        absolute_time = 0.0

        if verbose:
            print( "Video processing for " + video_worker.getWorkName() + " has begun" )

        #for timer...
        timer = TimeHelper()
        timer.startTimer()

        #open video...
        for video_idx, video_file in enumerate(self.file_list):
            try:
                capture = cv2.VideoCapture(video_file)
            except Exception as e:
                # error loading
                raise Exception( "The file <" + video_file + "> could not be opened")

            capture_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            capture_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # ...check size ...
            forced_resizing = False
            if width is None:
                # first video....
                # initialize local parameters....
                if self.forced_width is not None:
                    # ...size...
                    width = self.forced_width
                    height = self.forced_height

                    if capture_width != self.forced_width or capture_height != self.forced_height:
                        forced_resizing = True
                else:
                    width = capture_width
                    height = capture_height

                # on the worker class...
                video_worker.initialize(width, height)
            else:
                if self.forced_width is not None:
                    forced_resizing = (capture_width != self.forced_width or capture_height != self.forced_height)
                else:
                    if (width != capture_width) or (height != capture_height):
                        # invalid, all main video files must be the same resolution...
                        raise Exception("All video files on the list must have the same resolution")

            # get current FPS
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            # will use some frames per second
            jump_frames = int(video_fps / self.frames_per_second)

            # Read video until the end or until limit has been reached
            selection_step = 1
            timer_1 = TimeHelper()
            timer_2 = TimeHelper()

            while limit == 0 or offset_frame < limit:

                if selection_step == 2 or selection_step == 5:
                    # jump to frame in single step
                    timer_2.startTimer()

                    target_frame = capture.get(cv2.CAP_PROP_POS_FRAMES) + jump_frames - 1
                    valid_grab = capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                    timer_2.endTimer()

                    if selection_step == 2:
                        selection_step = 3

                if selection_step == 1 or selection_step == 4:
                    timer_1.startTimer()

                    # jump to frame by grabbing frames ...
                    valid_grab = True
                    for x in range(jump_frames - 1):
                        valid_grab = capture.grab()
                        if not valid_grab:
                            break

                    timer_1.endTimer()

                    if selection_step == 1:
                        selection_step = 2

                if selection_step == 3:
                    # decide which sampling grabbing method is faster
                    if timer_1.totalElapsedTime() < timer_2.totalElapsedTime():
                        print("Grabbing frames to jump")
                        selection_step = 4
                    else:
                        print("Jumping to frames directly")
                        selection_step = 5

                # get frame..
                if valid_grab:
                    flag, frame = capture.read()
                else:
                    flag, frame = False, None

                #print("Grab time: " + str(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                #print((valid_grab, flag, type(frame), selection_step, jump_frames))
                if not flag:
                    # end of video reached...
                    break
                else:                
                    offset_frame += 1
                    current_time = capture.get(cv2.CAP_PROP_POS_MSEC)
                    current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)

                if forced_resizing:
                    frame = cv2.resize(frame, (self.forced_width, self.forced_height))

                if offset_frame > 0:
                    frame_time = absolute_time + current_time
                    frame_idx = int(absolute_frame + current_frame)
                    video_worker.handleFrame(frame, last_frame, video_idx, frame_time, current_time, frame_idx)

                    if verbose and offset_frame % 50 == 0:
                        print( "Frames Processed = " + str(offset_frame) + \
                               ", Video Time = " + TimeHelper.stampToStr( frame_time ) )
                    
                    
                last_frame = frame
                last_time = current_time
                
            #at the end of the processing of current video
            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            video_length = capture.get(cv2.CAP_PROP_POS_MSEC)
            video_frames = capture.get(cv2.CAP_PROP_POS_FRAMES)
            
            absolute_time += video_length
            absolute_frame += video_frames

        #processing finished...
        video_worker.finalize()

        #end time counter...
        timer.endTimer()

        if verbose:
            print("Video processing for " + video_worker.getWorkName() + " completed: " +
                  TimeHelper.stampToStr(timer.lastElapsedTime() * 1000.0))


