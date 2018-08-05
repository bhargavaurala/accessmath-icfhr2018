#===================================================================
# Abstraction for Video Processing by segments. Use this class
# to extract samples of frames per segments of the video and apply
# specific algorithms to the batch of images extracted from each
# segment using an external class
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2015
#===================================================================

import math

import cv
import cv2

from AM_CommonTools.util.time_helper import TimeHelper


class VideoSegmentProcessor:
    # how to define the number of segments and their length
    SegmentTypeFixedTime = 0
    SegmentTypeFixedNumber = 1
    SegmentTypeCustomIntervals = 2

    # how many frames per segment
    FramesFixedFrameRate = 0
    FramesFixedNumber = 1

    # how to process frames per segment
    ProcessSequential = 0
    ProcessBulk = 1

    def __init__(self, file_list, segment_type, segments_val, frames_type, frames_val, process_type):
        self.file_list = file_list

        self.segment_type = segment_type
        self.segment_val = segments_val

        self.frames_type = frames_type
        self.frames_val = frames_val

        self.process_type = process_type

    def doProcessing(self, video_worker, limit=0, verbose=False):
        #initially....
        width = None
        height = None

        if verbose:
            print( "Video processing for " + video_worker.getWorkName() + " has begun" )

        # validate videos and compute total length ...
        total_length = 0.0
        all_video_lengths = []
        for video_idx, video_file in enumerate(self.file_list):
            try:
                capture = cv2.VideoCapture(video_file)
            except Exception as e:
                #error loading
                raise Exception( "The file <" + video_file + "> could not be opened")

            capture_width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
            capture_height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

            video_frames = capture.get(cv.CV_CAP_PROP_FRAME_COUNT)
            video_fps =  capture.get(cv.CV_CAP_PROP_FPS)

            capture.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 1)
            time_end = capture.get(cv.CV_CAP_PROP_POS_MSEC)

            video_length = time_end / 1000.0

            total_length += video_length
            all_video_lengths.append(video_length)

            #...check size ...
            if width == None:
                #first video....
                #initialize local parameters....
                #...size...
                width = capture_width
                height = capture_height

                if verbose:
                    print("Resolution is " + str(width) + " x " + str(height))

                #on the worker class...
                video_worker.initialize(width, height)
            else:
                if width != capture_width or height != capture_height:
                    #invalid, all main video files must be the same resolution...
                    raise Exception("All video files on the list must have the same resolution")

        # determine parts number and/or start/end times
        if self.segment_type == VideoSegmentProcessor.SegmentTypeFixedTime:
            # split by time ...
            n_parts = int(math.ceil(total_length / self.segment_val))


            intervals = [(self.segment_val * part, min(self.segment_val * (part + 1), total_length)) for part in range(n_parts)]
        elif self.segment_type == VideoSegmentProcessor.SegmentTypeFixedNumber:
            # fixed by number of parts
            n_parts = int(self.segment_val)

            part_length = total_length / n_parts

            intervals = [(part_length * part, part_length * (part + 1)) for part in range(n_parts)]
        elif self.segment_type == VideoSegmentProcessor.SegmentTypeCustomIntervals:
            # fixed by times ...
            # first, sort times, validate them and finally determine the parts ...
            if not isinstance(self.segment_val, list):
                raise Exception("Must especify a list of times")

            # sort and filter ....
            self.segment_val = [ time for time in sorted(self.segment_val) if time > 0.0 and time < total_length]

            n_parts = len(self.segment_val) + 1

            tempo_times = [0.0] + self.segment_val + [total_length]

            intervals = []
            for segment in range(n_parts):
                intervals.append((tempo_times[segment], tempo_times[segment + 1]))

        else:
            raise Exception("Invalid Segment Type")

        if verbose:
            print("Total video(s) length: " + TimeHelper.secondsToStr(total_length))
            print("Num. segments: " + str(n_parts))

        #for timer...
        timer = TimeHelper()
        timer.startTimer()

        # process every block
        current_video = -1
        capture = None

        global_time = 0.0
        local_time = 0.0
        for idx, times in enumerate(intervals):
            start_time, end_time = times

            # determine the sampling frame rate (based on time between frames) ...
            if self.frames_type == VideoSegmentProcessor.FramesFixedNumber:
                n_frames = int(self.frames_val)
                # time in ms
                sampling_time = ((end_time - start_time) / n_frames)
            elif self.frames_type == VideoSegmentProcessor.FramesFixedFrameRate:
                # time in ms
                sampling_time = 1.0 / float(self.frames_val)
            else:
                raise Exception("Invalid Frame Sampling mode")

            if verbose:
                print("Segment #" + str(idx + 1) + ": " + TimeHelper.secondsToStr(start_time) + " to " + TimeHelper.secondsToStr(end_time) + " (Dist: " + str(sampling_time) + " s)")
                #print("Sampling frame each " + str(sampling_time) + " s")

            frame_count = 0
            block_frames = []
            next_frame_time = start_time

            while next_frame_time < end_time:
                # check if current video contains the sample ... load next video if not ...
                while capture is None or (global_time + all_video_lengths[current_video] < next_frame_time):
                    # open next video ...
                    if current_video >= 0:
                        global_time += all_video_lengths[current_video]

                    current_video += 1
                    video_file = self.file_list[current_video]

                    try:
                        if verbose:
                            print("Loading: " + video_file)

                        capture = cv2.VideoCapture(video_file)
                    except Exception as e:
                        #error loading
                        raise Exception( "The file <" + video_file + "> could not be opened")

                    # get the first frame ...
                    flag, last_frame = capture.read()
                    if not flag:
                        raise Exception("Error found while reading file <" + video_length + "> ")
                    else:
                        local_time = capture.get(cv.CV_CAP_PROP_POS_MSEC) / 1000.0
                        frame_length = local_time / 1000.0

                # get the desired frame ...
                local_target = next_frame_time - global_time

                if local_target < local_time:
                    # reuse the last frame read ...
                    current_frame = last_frame
                else:
                    # first jump undesired frames ...
                    tempo_diffs = []
                    while local_time + frame_length < local_target:
                        # jump to next frame ...
                        capture.grab()
                        new_time = capture.get(cv.CV_CAP_PROP_POS_MSEC) / 1000.0
                        frame_length = new_time - local_time
                        local_time = new_time

                    # read the desired frame
                    flag, current_frame = capture.read()
                    local_time += frame_length
                    if not flag:
                        print(local_time)
                        print(global_time)
                        print(all_video_lengths)
                        print(frame_length)
                        print(next_frame_time)
                        print(local_target)
                        print(current_video)
                        print("---")
                        print(capture.get(cv.CV_CAP_PROP_POS_MSEC))
                        print(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
                        print(capture.get(cv.CV_CAP_PROP_FPS))
                        print(tempo_diffs)
                        raise Exception("Failure")

                # process frame ...
                if self.process_type == VideoSegmentProcessor.ProcessSequential:
                    # process images one by one ...
                    if frame_count > 0:
                        video_worker.handleFrame(current_frame, last_frame, idx, current_video,
                                                 next_frame_time, local_target)

                    if verbose and (frame_count + 1) % 50 == 0:
                        print( "Frames Processed = " + str(frame_count) +
                               ", Video Time = " + TimeHelper.secondsToStr(next_frame_time))
                elif self.process_type == VideoSegmentProcessor.ProcessBulk:
                    # save to process later ...
                    block_frames.append(current_frame)
                else:
                    raise Exception("Invalid process mode selected")

                # move to next
                next_frame_time += sampling_time
                last_frame = current_frame

                frame_count += 1
                if self.frames_type == VideoSegmentProcessor.FramesFixedNumber and frame_count >= int(self.frames_val):
                    # end current segment ...
                    break

            # if process is done by block, process them now ...
            if self.process_type == VideoSegmentProcessor.ProcessBulk:
                if verbose:
                    print("... Processsing bulk of sampled images ... ")
                video_worker.handleBlock(block_frames, idx, start_time, end_time)




