import os
import cv2

class DeepRoiDetection(object):
    """
    This is the DRoiD you are looking for!
    Base interface for Deep Neural Network based region of interest detection and video_worker.
    This module supports bounding boxes in the xyxy or xywh formats. We set default as xyxy for AccessMath.
    """
    def __init__(self,
                 net,
                 name = 'DRoiD',
                 detection_threshold=0.6,
                 bbox_format='xyxy'):
        self.name = name
        self.net = net # use a util.model_loader object to pass a model here
        self.detected_rois = {}
        self.detection_threshold = detection_threshold
        self.bbox_format = 'xyxy' if bbox_format not in ['xyxy', 'xywh'] else bbox_format

    def initialize(self, width, height):
        pass

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = '{}_{}_debug({})'.format(video_name, self.name, self.detection_threshold)
        self.debug_video_name = video_name
        if not os.path.isdir(self.debug_out_dir) and self.debug_mode:
            os.makedirs(self.debug_out_dir)

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        img = frame
        frameID = abs_frame_idx
        H, W, C = img.shape
        self.detected_rois[frameID] = {}
        self.detected_rois[frameID]['abs_time'] = abs_time
        self.detected_rois[frameID]['height'] = H
        self.detected_rois[frameID]['width'] = W
        self.detected_rois[frameID]['bboxes'] = bboxes = []
        self.detected_rois[frameID]['labels'] = labels = []
        self.detected_rois[frameID]['confidences'] = confidences = []
        self.detected_rois[frameID]['features'] = features = []

        self.fprop(img, bboxes, labels, confidences, features)

        self.detected_rois[frameID]['bboxes'] = [bbox for i, bbox in enumerate(bboxes)\
                                                 if confidences[i] >= self.detection_threshold]
        self.detected_rois[frameID]['labels'] = [label for i, label in enumerate(labels)\
                                                 if confidences[i] >= self.detection_threshold]
        self.detected_rois[frameID]['confidences'] = [conf for conf in confidences\
                                                      if conf >= self.detection_threshold]
        self.detected_rois[frameID]['visible'] = len(self.detected_rois[frameID]['bboxes']) > 0

        if self.debug_mode:
            self.debug_frame(img, frameID)

    def fprop(self, img, bboxes, labels, confidences, features):
        """
        This function needs to be implemented according the neural network architecture you have chosen.
        We expect the derived classes to implement this method based on the model of choice and append to
        bboxes, labels and confidences arrays, *the same ones passed as args*
        If your model returns an RoI which is not bounding box in self.format then it should be converted so.
        If your model returns only one kind of class append a string that describes it for each bbox.
        If your model does not return a confidence append 1.0 for each bbox.

        :param img: Input frame as np array in BGR channel ordering
        :param bboxes: List of xyxy or xywh tuples detections in the image, to be appended in implementation
        :param labels: Corresponding list of xyxy or xywh tuples labels, to be appended in implementation
        :param confidences: Corresponding list of xyxy or xywh tuples confidences, to be appended in implementation
        :param features: Corresponding list of feature vectors describing the roi, to be appended in implementation
        """
        raise NotImplementedError

    def debug_frame(self, frame, frameID):
        bboxes = self.detected_rois[frameID]['bboxes']
        for i, bbox in enumerate(bboxes):
            score = self.detected_rois['confidences'][i]
            label = self.detected_rois['labels'][i]
            if score >= self.detection_threshold:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0))
                cv2.putText(frame, str(label) + ' ' + str(score), (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.imwrite('{}/{}.jpg'.format(self.debug_out_dir, frameID), frame)

    def getWorkName(self):
        return self.name

    def finalize(self):
        pass

    def get_results(self):
        return self.detected_rois

    @staticmethod
    def getCentroid(bbox, bbox_format):
        if bbox_format == 'xyxy':
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
        else:
            x, y, w, h = bbox
        return (x + (w // 2), y + (h // 2))

    @staticmethod
    def getArea(bbox, bbox_format):
        if bbox_format == 'xyxy':
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
        else:
            x, y, w, h = bbox
        return w * h














