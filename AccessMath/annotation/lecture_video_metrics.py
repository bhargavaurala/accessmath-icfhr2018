
import datetime
import cv2

from .video_metrics import VideoMetrics

class LectureVideosMetrics:
    def __init__(self):
        self.metrics_per_video = {}

    def process_lecture(self, process, input_data):
        lecture = process.current_lecture
        m_videos = [video["path"] for video in lecture.main_videos]

        # print(m_videos)
        opencv_metrics = []
        for idx, video_filename in enumerate(m_videos):
            # print(video_filename)
            # for debug only, get OpenCV capture metrics ...
            capture = cv2.VideoCapture(video_filename)
            prop_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            prop_end_frame_count = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            prop_end_frame_time = capture.get(cv2.CAP_PROP_POS_MSEC)

            opencv_metrics.append(
                {
                    "frame_count": prop_frame_count,
                    "end_frame": prop_end_frame_count,
                    "end_time": prop_end_frame_time,
                }
            )

        print("OpenCV Property Metrics")
        print("V.IDX\t# Fr\tEnd F\tEnd T")
        for idx, video_filename in enumerate(m_videos):
            vid_metrics = opencv_metrics[idx]
            length_str = str(datetime.timedelta(milliseconds=vid_metrics["end_time"]))
            print("{0:d}\t{1:d}\t{2:d}\t{3:s}".format(idx + 1, vid_metrics["frame_count"], vid_metrics["end_frame"], length_str))

        print(".... counting frames ....")
        video_metrics = VideoMetrics.FromVideoFiles(m_videos)
        print("")
        print("Count by grabbing all metrics:")
        video_metrics.print_metrics()

        self.metrics_per_video[lecture.title] = video_metrics

    def to_dict(self):
        result = {}

        for lecture_title in self.metrics_per_video:
            result[lecture_title] = self.metrics_per_video[lecture_title].to_dict()

        return result
