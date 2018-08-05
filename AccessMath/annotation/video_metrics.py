
import datetime

import cv2

class VideoMetrics:
    def __init__(self):
        self.total_frames = 0
        self.total_time = 0.0
        self.per_video_last_frame = []
        self.per_video_frames = []
        self.per_video_time = []
        self.video_files = []

    def print_metrics(self):
        print("V.IDX\tGrabs\tEnd F\tEnd T")
        for idx, video_filename in enumerate(self.video_files):
            total_grabs = self.per_video_frames[idx]
            end_frame = self.per_video_last_frame[idx]
            end_time = self.per_video_time[idx]
            length_str = str(datetime.timedelta(milliseconds=end_time))

            print("{0:d}\t{1:d}\t{2:d}\t{3:s}".format(idx + 1, total_grabs, end_frame, length_str))

    def to_dict(self):
        return {
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "per_video_last_frame": self.per_video_last_frame,
            "per_video_frames": self.per_video_frames,
            "per_video_time": self.per_video_time,
            "video_files": self.video_files,
        }

    @staticmethod
    def FromVideoFiles(video_files):
        metrics = VideoMetrics()
        metrics.video_files = video_files

        for idx, video_filename in enumerate(video_files):
            # print(video_filename)
            # for debug only, get OpenCV capture metrics ...
            capture = cv2.VideoCapture(video_filename)

            flag = True
            total_grabs = 0
            while flag:
                flag = capture.grab()
                if flag:
                    total_grabs += 1

            end_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            end_time = capture.get(cv2.CAP_PROP_POS_MSEC)

            metrics.per_video_frames.append(total_grabs)
            metrics.per_video_last_frame.append(end_frame)
            metrics.per_video_time.append(end_time)
            metrics.total_frames += total_grabs
            metrics.total_time += end_time

        return metrics
