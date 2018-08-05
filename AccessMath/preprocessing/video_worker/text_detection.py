import cv2
import skimage
import numpy as np
import caffe

import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint

from deep_roi_detection import DeepRoiDetection

def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    # polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def nms(boxes, threshold):
    # print 'boxes',boxes
    nms_flag = [True] * len(boxes)

    for i, b in enumerate(boxes):
        if not nms_flag[i]:
            continue
        else:
            for j, a in enumerate(boxes):
                if a == b:
                    continue
                if not nms_flag[j]:
                    continue
                rec1 = b[0:8]
                rec2 = a[0:8]
                polygon_points1 = np.array(rec1).reshape(4, 2)
                poly1 = Polygon(polygon_points1)
                polygon_points2 = np.array(rec2).reshape(4, 2)
                poly2 = Polygon(polygon_points2)
                if poly1.area == 0:
                    nms_flag[i] = False
                    continue
                if poly2.area == 0:
                    nms_flag[j] = False
                    continue
                iou = polygon_iou(rec1, rec2)
                if iou > threshold:
                    if b[8] > a[8]:
                        nms_flag[j] = False
                    elif b[8] == a[8] and poly1.area > poly2.area:
                        nms_flag[j] = False
                    elif b[8] == a[8] and poly1.area <= poly2.area:
                        nms_flag[i] = False
                        break
    return nms_flag

caffe.set_device(0)
caffe.set_mode_gpu()

default_scales = ((320, 320), (640, 480), (640, 640), (960, 540), (1920, 1080))
default_feature_layers = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'global']

class TextDetection(DeepRoiDetection):
    def __init__(self,
                 net,
                 name='HCDroiD',
                 detection_threshold=0.6,
                 nms_threshold=0.3,
                 bbox_format='xyxy',
                 scales=default_scales):
        DeepRoiDetection.__init__(self, net, name, detection_threshold, bbox_format)
        self.nms_threshold = nms_threshold
        self.scales = scales

    def fprop(self, img, bboxes, labels, confidences, features):
        raw_image = img
        dt_results = []
        for scale in self.scales:
            transformed_image = self.convertImageCaffe(raw_image, scale)
            self.net.blobs['data'].reshape(1, 3, scale[0], scale[1])
            self.net.blobs['data'].data[...] = transformed_image
            # Forward pass.
            dets = self.net.forward()['detection_out']
            # Parse the outputs.
            det_label = dets[0, 0, :, 1]
            det_conf = dets[0, 0, :, 2]
            det_xmin = dets[0, 0, :, 3]
            det_ymin = dets[0, 0, :, 4]
            det_xmax = dets[0, 0, :, 5]
            det_ymax = dets[0, 0, :, 6]
            top_indices = range(len(det_conf))
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * raw_image.shape[1]))
                ymin = int(round(top_ymin[i] * raw_image.shape[0]))
                xmax = int(round(top_xmax[i] * raw_image.shape[1]))
                ymax = int(round(top_ymax[i] * raw_image.shape[0]))
                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(raw_image.shape[1] - 1, xmax)
                ymax = min(raw_image.shape[0] - 1, ymax)
                score = top_conf[i]
                dt_result = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, score]
                dt_results.append(dt_result)
        dt_results = sorted(dt_results, key=lambda x: -float(x[8]))
        nms_flag = nms(dt_results, self.nms_threshold)
        for k, dt in enumerate(dt_results):
            if nms_flag[k]:
                xmin = dt[0]
                ymin = dt[1]
                xmax = dt[2]
                ymax = dt[5]
                score = dt[8]
                bboxes += [(xmin, ymin, xmax, ymax)]
                labels += ['HC']
                confidences += [score]

    @staticmethod
    def convertImageCaffe(raw_image, scale):
        # convert raw_image to RGB and float32 before passing as input to the network
        # using skimage.img_as_float seems to be the best way to encode before passing to the network
        # do not pass opencv array directly! even after converting channel order!!
        input_img = skimage.img_as_float(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        transformed_image = transformer.preprocess('data', input_img)
        return transformed_image


class TextDetectionFeatures(DeepRoiDetection):
    def __init__(self,
                 net,
                 detected_rois,
                 name='HCDroiDFeat',
                 detection_threshold=0.6,
                 nms_threshold=0.3,
                 bbox_format='xyxy',
                 scales=default_scales):
        DeepRoiDetection.__init__(self, net, name, detection_threshold, bbox_format)
        self.nms_threshold = nms_threshold
        self.scales = scales
        self.detected_rois = detected_rois


    def fprop(self, img, bboxes, labels, confidences, features):
        raw_image = img
        img_feat = []
        for scale in self.scales:
            transformed_image = self.convertImageCaffe(raw_image, scale)
            self.net.blobs['data'].reshape(1, 3, scale[0], scale[1])
            self.net.blobs['data'].data[...] = transformed_image
            # Forward pass.
            self.net.forward()


    @staticmethod
    def convertImageCaffe(raw_image, scale):
        # convert raw_image to RGB and float32 before passing as input to the network
        # using skimage.img_as_float seems to be the best way to encode before passing to the network
        # do not pass opencv array directly! even after converting channel order!!
        input_img = skimage.img_as_float(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        transformed_image = transformer.preprocess('data', input_img)
        return transformed_image
