import torch
from torch.autograd import Variable
from data import BaseTransform, VOC_CLASSES as labelmap

from deep_roi_detection import DeepRoiDetection

class PersonDetection(DeepRoiDetection):
    def __init__(self,
                 net,
                 name = 'PersonDRoiD',
                 detection_threshold=0.6,
                 bbox_format='xyxy'):
        DeepRoiDetection.__init__(self, net, name, detection_threshold, bbox_format)
        self.transform = BaseTransform(self.net.size, (104, 117, 123))

    def fprop(self, img, bboxes, labels, confidences, features):
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        x = x.cuda()
        y = self.net(x)
        detections = y.data
        # scale each detection back to original image shape
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        for i in range(detections.size(1)):
            if labelmap[i - 1] is not 'person':
                continue
            for j in range(detections.size(2)):
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                if self.bbox_format == 'xywh':
                    coords = (pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1])
                else:
                    coords = (pt[0], pt[1], pt[2], pt[3])
                bboxes += [coords]
                labels += ['person']
                confidences += [score]
