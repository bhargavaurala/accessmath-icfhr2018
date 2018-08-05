import json
import numpy as np
import cv2

from AM_CommonTools.util.time_helper import TimeHelper

class ImageListGenerator(object):
    def __init__(self, folder, extension, preload=False):
        self.folder = folder
        self.im_ext = extension
        self.index_path = '{}/index.json'.format(self.folder)
        print(self.folder)
        with open(self.index_path, 'r') as f:
            self.metadata = json.load(f)
        self.metadata['0'] = {'video_time': 0.0,
                              'frame_idx': 0,
                              'abs_time': 0.0,
                              'video_idx': 0}
        self.frameIDs = map(int, self.metadata.keys())
        self.frameIDs.sort()
        print('Number of Frames', len(self.frameIDs) - 1)
        im = cv2.imread('{}/{}.{}'.format(self.folder, self.frameIDs[1], self.im_ext))
        self.height, self.width, self.channels = im.shape if im is not None else (None, None, None)
        print(self.width, self.height)
        self.curr_idx = 0
        self.properties = self.metadata[str(self.frameIDs[0])].keys()
        self.preload = preload
        if self.preload:
            self.ims = np.empty((len(self.frameIDs), self.height, self.width, self.channels), dtype=np.uint8)
            print('preloading images...')
            for i, frameID in enumerate(self.frameIDs[1:]):
                self.ims[i, ...] = cv2.imread('{}/{}.{}'.format(self.folder, frameID, self.im_ext))
            print('done')
        else:
            self.ims = []
            for i, frameID in enumerate(self.frameIDs[1:]):
                self.ims += ['{}/{}.{}'.format(self.folder, frameID, self.im_ext)]

    def __len__(self):
        return len(self.frameIDs) - 1

    def __getitem__(self, item):
        try:
            if self.preload:
                return self.ims[item]
            else:
                im = cv2.imread(self.ims[item])
                return im
        except Exception as e:
            print(e, item, len(self.frameIDs), len(self))

    def refresh(self):
        self.curr_idx = 0 if self.curr_idx == -1 else self.curr_idx

    def index2frameID(self):
        self.curr_idx = -1 if self.curr_idx >= len(self) else self.curr_idx
        return self.frameIDs[self.curr_idx]

    def read(self):
        # print('reading {}'.format(self.curr_idx))
        if self.curr_idx >= len(self) or self.curr_idx < 0:
            return False, None
        frame = self[self.curr_idx]
        self.curr_idx += 1
        return True, frame

    def get(self, prop):
        if prop not in self.properties:
            return None
        self.curr_idx = -1 if self.curr_idx >= len(self) else self.curr_idx
        return self.metadata[str(self.frameIDs[self.curr_idx])][prop]

class ImageListProcessor:
    def __init__(self, src_dir, frames_per_second=-1):
        self.src_dir = src_dir
        self.frames_per_second = frames_per_second
        self.forced_width = None
        self.forced_height = None

    def force_resolution(self, width, height):
        self.forced_width = width
        self.forced_height = height

    def checkError(self):
        print("Works?")

    def doProcessing(self, video_worker, limit=0, verbose=False):
        # initially....
        width = None
        height = None

        offset_frame = -1
        absolute_frame = 0
        absolute_time = 0.0

        if verbose:
            print("Video processing for " + video_worker.getWorkName() + " has begun")

        # for timer...
        timer = TimeHelper()
        timer.startTimer()

        # open video...
        try:
            print(self.src_dir)
            capture =  ImageListGenerator('{}/{}'.format(self.src_dir, 'JPEGImages'), 'jpg')
        except Exception as e:
            # error loading
            print(e)
            raise Exception("The directory <" + self.src_dir + "> is not in the correct export format, check index.json")

        last_frame = None
        capture_width = capture.width
        capture_height = capture.height
        # print(capture_width, capture_height)

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
                    raise Exception("All files on the list must have the same resolution")

        while limit == 0 or offset_frame < limit:
            # get frame..
            flag, frame = capture.read()
            if not flag:
                # end of video reached...
                print('end of video reached...')
                break
            else:
                offset_frame += 1
                current_time = capture.get('abs_time')
                current_frame = capture.index2frameID()
            if forced_resizing:
                frame = cv2.resize(frame, (self.forced_width, self.forced_height))

            if offset_frame >= 0:
                frame_time = absolute_time + current_time
                frame_idx = int(absolute_frame + current_frame)
                video_worker.handleFrame(frame, last_frame, 0, frame_time, current_time, frame_idx)

                if verbose and offset_frame % 50 == 0:
                    print("Frames Processed = " + str(offset_frame) +
                          ", Video Time = " + TimeHelper.stampToStr(frame_time))

            last_frame = frame
            last_time = current_time

        # processing finished...
        video_worker.finalize()

        # end time counter...
        timer.endTimer()

        if verbose:
            print("Video processing for " + video_worker.getWorkName() + " completed: " +
                  TimeHelper.stampToStr(timer.lastElapsedTime() * 1000.0))