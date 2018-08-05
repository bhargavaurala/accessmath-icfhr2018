
import time
import cv2
import numpy as np

class OpenCVVideoPlayer:
    FrameCache = 500 # around 3 GB (500, un-compressed)

    def __init__(self, video_files, forced_resolution=None):
        self.current_time = None
        self.last_time = time.time()

        if len(video_files) < 1:
            raise Exception("Must specify at least one video file")

        self.video_files = video_files

        self.width = None
        self.height = None

        self.playing = False
        self.play_abs_position = 0.0
        self.play_video = 0
        self.play_speed = 1.0
        self.play_capture = None
        self.end_reached = False

        if forced_resolution is None:
            self.forced_width, self.forced_height = None, None
        else:
            self.forced_width, self.forced_height = forced_resolution

        self.total_frames = 0
        self.total_length = 0
        self.video_lengths = []
        self.video_offsets = [0.0]
        self.frame_offsets = [0]
        self.video_frames = []
        self.last_frame_img = None
        self.last_frame_idx = None

        self.cache_images = []
        self.cache_times = []
        self.cache_frames = []
        self.cache_offset = 0.0
        self.cache_pos = 0

        # get some basic properties
        # total length, frames
        for video_idx, video_file in enumerate(self.video_files):
            try:
                capture = cv2.VideoCapture(video_file)
            except Exception as e:
                #error loading
                raise Exception( "The file <" + video_file + "> could not be opened")

            if self.forced_width is None:
                capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                capture_width = self.forced_width
                capture_height = self.forced_height

            if self.width is None:
                self.width = capture_width
                self.height = capture_height

                flag, self.last_frame_img = capture.read()

                dummy_get = capture.get(cv2.CAP_PROP_POS_MSEC)

                """
                self.cache_images.append(self.last_frame)
                self.cache_times.append(next_time)
                self.cache_frames.append(0)
                self.cache_offset = 0.0
                """
            else:
                if self.width != capture_width or self.height != capture_height:
                    raise Exception( "The resolutions of the specified video files do not match")

            # video_fps =  capture.get(cv.CV_CAP_PROP_FPS)
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            time_end = capture.get(cv2.CAP_PROP_POS_MSEC)
            # print((video_idx, time_end, video_frames, capture.get(cv2.CAP_PROP_POS_FRAMES)))

            video_length = time_end

            self.video_lengths.append(video_length)
            self.video_frames.append(video_frames)

            self.total_length += video_length
            self.total_frames += video_frames

            self.video_offsets.append(self.total_length)
            self.frame_offsets.append(self.frame_offsets[-1] + video_frames)

        self.current_capture = None

        self.black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.frame_changed_callback = None

    def play(self):
        self.current_time = None
        self.last_time = time.time()

        self.playing = True
        self.end_reached = False

    def pause(self):
        self.playing = False

    def set_position_time(self, new_abs_position):
        # check if cache has to be cleaned (new position not in cache)
        raise Exception("Not implemented")

    def set_position_frame(self, new_abs_frame, notify_listeners):
        if new_abs_frame >= self.frame_offsets[-1]:
            new_abs_frame = self.frame_offsets[-1] - 2

        if new_abs_frame < 0:
            new_abs_frame = 0

        # first, check if the frame is still in the cache ...
        if len(self.cache_frames) > 0 and (self.cache_frames[0] <= new_abs_frame <= self.cache_frames[-1]):
            # already on cache, just move the time to the right frame
            offset = new_abs_frame - self.cache_frames[0]
            # adjust
            if offset > 0:
                self.play_abs_position = self.cache_times[offset - 1]
            else:
                # first element on cache ...
                self.play_abs_position = self.cache_offset

            self.cache_pos = offset
            self.last_frame_img = self.cache_images[self.cache_pos]
            self.last_frame_idx = self.cache_frames[self.cache_pos]
        else:
            # reset cache
            self.cache_frames = []
            self.cache_images = []
            self.cache_times = []
            self.cache_pos = 0

            # find desired video
            opened_video = self.play_video if self.current_capture is not None else -1
            self.play_video = 0

            while new_abs_frame > self.frame_offsets[self.play_video + 1]:
                self.play_video += 1

            #print("Jumping to video offset " + str(self.play_video))
            # open the video at desired frame
            offset = new_abs_frame - self.frame_offsets[self.play_video]
            if opened_video != self.play_video:
                self.current_capture = cv2.VideoCapture(self.video_files[self.play_video])

            self.current_capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
            # update video location and cache start
            self.play_abs_position = self.current_capture.get(cv2.CAP_PROP_POS_MSEC) + self.video_offsets[self.play_video]
            self.cache_offset = self.play_abs_position

            #print("here")

            # read next frame
            self.last_frame_img, self.last_frame_idx = self.__extract_next_frame()

            #print("done")

        if self.frame_changed_callback is not None and notify_listeners:
            self.frame_changed_callback(int(new_abs_frame), self.play_abs_position)

        self.last_time = time.time()

    def __extract_next_frame(self):
        if self.current_capture is None:
            self.play_video = 0
            self.current_capture = cv2.VideoCapture(self.video_files[self.play_video])
            self.play_abs_position = 0.0

        # get the next frame ...
        flag = False
        while not flag:
            flag, next_frame = self.current_capture.read()
            if self.forced_width is not None and flag:
                next_frame = cv2.resize(next_frame, (self.forced_width, self.forced_height))

            if not flag:
                # failed to get frame from current capture... try opening next ...
                self.play_video += 1

                if self.play_video < len(self.video_files):
                    # open next
                    self.current_capture = cv2.VideoCapture(self.video_files[self.play_video])

                    flag, next_frame = self.current_capture.read()
                    if self.forced_width is not None and flag:
                        next_frame = cv2.resize(next_frame, (self.forced_width, self.forced_height))
                else:
                    self.end_reached = True
                    next_frame = None
                    break

        if next_frame is not None:
            # update cache ...
            last_time = self.current_capture.get(cv2.CAP_PROP_POS_MSEC) + self.video_offsets[self.play_video]
            frame_number = int(self.current_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1 + self.frame_offsets[self.play_video]

            self.cache_times.append(last_time)
            self.cache_images.append(next_frame)
            self.cache_frames.append(frame_number)

            while len(self.cache_times) > OpenCVVideoPlayer.FrameCache:
                self.cache_offset = self.cache_times[0]
                if self.cache_pos > 0:
                    self.cache_pos -= 1
                del self.cache_times[0]
                del self.cache_images[0]
                del self.cache_frames[0]
        else:
            frame_number = self.total_frames

        return next_frame, frame_number


    def __frame_in_cache(self, abs_time):
        if len(self.cache_times) == 0:
            return False

        return self.cache_offset <= abs_time <= self.cache_times[-1]


    def __get_cached_frame(self, abs_time):
        if self.end_reached:
            if len(self.cache_images) > 0:
                return self.cache_images[-1], self.cache_frames[-1]
            else:
                return self.black_frame, self.total_frames

        # assume forward movement only
        while self.cache_times[self.cache_pos] < abs_time:
            self.cache_pos += 1

        return self.cache_images[self.cache_pos], self.cache_frames[self.cache_pos]

    def get_frame(self):
        self.current_time = time.time()
        # in seconds ...
        delta = (self.current_time - self.last_time) * self.play_speed
        self.last_time = self.current_time
        #print(1.0 / delta)

        if self.playing:
            # update last frame ...
            # use milliseconds (just like opencv does)
            self.play_abs_position += delta * 1000.0

            # while desired frame not in cache ....
            while not self.__frame_in_cache(self.play_abs_position) and not self.end_reached:
                # ... update cache
                self.__extract_next_frame()

            # get frame
            self.last_frame_img, self.last_frame_idx = self.__get_cached_frame(self.play_abs_position)

            if self.frame_changed_callback is not None:
                if len(self.cache_frames) > 0:
                    self.frame_changed_callback(int(self.cache_frames[self.cache_pos]), self.cache_times[self.cache_pos])

        return self.last_frame_img, self.last_frame_idx

    def update_video_metrics(self, video_metrics):
        self.video_lengths = video_metrics["per_video_time"]
        self.video_frames = video_metrics["per_video_frames"]
        self.total_length = video_metrics["total_time"]
        self.total_frames = video_metrics["total_frames"]
        self.video_offsets = [0.0]
        self.frame_offsets = [0]

        for idx in range(len(self.video_frames)):
            self.video_offsets.append(self.video_offsets[-1] + self.video_lengths[idx])
            self.frame_offsets.append(self.frame_offsets[-1] + self.video_frames[idx])

