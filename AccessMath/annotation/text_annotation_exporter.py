
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .video_object import VideoObject
from .video_object_location import VideoObjectLocation

class TextAnnotationExporter:
    ExportModeAllPerFrame = 0
    ExportModeUniqueBoxes = 1

    def __init__(self, export_mode,  video_objects, canvas_loc, render_loc, render_size, export_dir):
        self.export_mode = export_mode
        self.img_width = None
        self.img_height = None

        # Source render info ...
        self.canvas_loc = canvas_loc
        self.render_loc = render_loc
        self.render_size = render_size

        self.proj_off_x = None
        self.proj_off_y = None
        self.proj_scale_x = None
        self.proj_scale_y = None

        # directory where results will be stored ...
        self.export_dir = export_dir
        self.export_img_dir = export_dir + "/images"
        self.export_xml_dir = export_dir + "/xml"

        self.video_objects = video_objects

        self.text_objects = []
        self.speaker = None

        # for unique-objects export mode
        self.exported_text_objects = None
        self.unique_objects_xml_tree = None

        # filter text annotations
        for video_object in self.video_objects:
            if TextAnnotationExporter.CheckTextObject(video_object):
                # a text region object found ...
                self.text_objects.append(video_object)
            else:
                if video_object.id.lower() == "speaker":
                    # speaker object found ...
                    self.speaker = video_object


    def initialize(self, width, height, prepare_dirs=True):
        self.img_width = width
        self.img_height = height

        # ... projection info ...
        render_x, render_y = self.render_loc
        render_w, render_h = self.render_size
        canvas_x, canvas_y = self.canvas_loc

        self.proj_off_x = render_x - canvas_x
        self.proj_off_y = render_y - canvas_y
        # note that these values should be the same (if aspect ratio is kept)
        self.proj_scale_x = self.img_width / render_w
        self.proj_scale_y = self.img_height / render_h

        if self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            self.exported_text_objects = {}
            self.unique_objects_xml_tree = ET.Element('annotation')

        # prepare export root ...
        if prepare_dirs:
            os.makedirs(self.export_img_dir, exist_ok=True)
            os.makedirs(self.export_xml_dir, exist_ok=True)

    def getWorkName(self):
        return "Text Annotation Exporter"

    def project_object_location(self, loc):
        proj_x = (loc.x - self.proj_off_x) * self.proj_scale_x
        proj_y = (loc.y - self.proj_off_y) * self.proj_scale_y
        proj_w = loc.w * self.proj_scale_x
        proj_h = loc.h * self.proj_scale_y

        proj_loc = VideoObjectLocation(loc.visible, loc.frame, loc.abs_time, proj_x, proj_y, proj_w, proj_h)

        return proj_loc

    def frame_visible_bboxes_state(self, frame_idx):
        # find speaker location
        if self.speaker is None:
            speaker_loc = None
        else:
            speaker_loc = self.speaker.get_location_at(frame_idx, False)

        # for each text object ...
        not_occluded_bboxes = []
        occluded_bboxes = []
        for text_object in self.text_objects:
            # get interpolated location at current frame (if present)
            text_loc = text_object.get_location_at(frame_idx, False)
            text_name = text_object.name

            # check if text box is present on current frame ..
            if text_loc is not None and text_loc.visible:
                # text is in the current frame ...

                # check if occluded by the speaker
                if (speaker_loc is None) or (not speaker_loc.visible):
                    # no speaker on the image
                    # iou = 0.0
                    int_area_prc = 0.0
                else:
                    # speaker is on the image, check for occlusion as defined by IOU
                    # iou = speaker_loc.IOU(text_loc)
                    int_area_prc = text_loc.intersection_percentage(speaker_loc)

                # project ...
                proj_loc = self.project_object_location(text_loc)

                # mark as either occluded or not ...
                # if iou < 0.0001:
                # TODO: this threhsold should come from config ...
                if int_area_prc < 0.25:
                    # not enough intersection with speaker ...
                    not_occluded_bboxes.append((text_name, proj_loc.get_XYXY_box()))
                else:
                    # intersects with the speaker too much, deemed as occluded!
                    occluded_bboxes.append((text_name, proj_loc.get_XYXY_box()))

        return speaker_loc, not_occluded_bboxes, occluded_bboxes

    def debug_show_bboxes(self, frame, frame_idx, speaker_loc, not_occluded_bboxes, occluded_bboxes):
        frame_copy = frame.copy()

        for text_name, (x1, y1, x2, y2) in not_occluded_bboxes:
            pt1 = int(x1), int(y1)
            pt2 = int(x2), int(y2)
            cv2.rectangle(frame_copy, pt1, pt2, (0, 255, 0), thickness=2)

        for text_name, (x1, y1, x2, y2) in occluded_bboxes:
            pt1 = int(x1), int(y1)
            pt2 = int(x2), int(y2)
            cv2.rectangle(frame_copy, pt1, pt2, (0, 0, 255), thickness=2)

        if speaker_loc is not None and speaker_loc.visible:
            proj_speaker = self.project_object_location(speaker_loc)
            x1, y1, x2, y2 = proj_speaker.get_XYXY_box()

            pt1 = int(x1), int(y1)
            pt2 = int(x2), int(y2)
            cv2.rectangle(frame_copy, pt1, pt2, (255, 0, 0), thickness=2)

        debug_filename = "{0:s}/debug/{1:d}.png".format(self.export_dir, frame_idx)
        cv2.imwrite(debug_filename, frame_copy)

    def export_all_by_frame(self, frame, frame_idx, not_occluded_bboxes):
        # Output file names ...
        out_img_filename = "{0:s}/{1:d}.png".format(self.export_img_dir, frame_idx)
        out_xml_filename = "{0:s}/{1:d}.xml".format(self.export_xml_dir, frame_idx)

        # Export Bounding Boxes in XML format ...
        # ... get XML for non-occluded boxes
        # ... Save boxes ...
        xml_tree = TextAnnotationExporter.generate_XML_objects(out_img_filename, self.img_width, self.img_height,
                                                               not_occluded_bboxes)
        xml_tree.write(out_xml_filename)

        # ... save image ...
        cv2.imwrite(out_img_filename, frame)

    def export_unique_objects(self, frame, frame_idx, not_occluded_bboxes):
        # check which objects are initially visible and can be exported (not exported before)
        for text_name, bbox in not_occluded_bboxes:
            x1, y1, x2, y2 = bbox
            # ... crop out of boundaries ...
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(int(self.img_width), int(x2))
            y2 = min(int(self.img_height), int(y2))

            bbox = x1, y1, x2, y2

            # print((text_name, frame_idx, bbox))

            region_img = frame[y1:y2, x1:x2]
            current_object = (frame_idx, bbox, region_img)

            if text_name not in self.exported_text_objects:
                # mark as exported ...
                self.exported_text_objects[text_name] = [current_object]

                out_img_filename = "{0:s}/{1:s}.png".format(self.export_img_dir, text_name)

                # ... save xml ...
                self.append_XML_unique_object(out_img_filename, text_name, bbox)

            else:
                self.exported_text_objects[text_name].append(current_object)

    def handleFrame(self, frame, last_frame, video_idx, frame_time, current_time, frame_idx):
        # Compute and export sample frame metadata
        speaker_loc, not_occluded_bboxes, occluded_bboxes = self.frame_visible_bboxes_state(frame_idx)

        # total_in_frame = len(occluded_boxes) + len(not_occluded_bboxes)
        # print("-> Text count: {0:d} / {1:d}".format(len(not_occluded), total_in_frame))

        if self.export_mode == TextAnnotationExporter.ExportModeAllPerFrame:
            # export the frame (single image) and a file with all the metadata for all GT bboxes
            self.export_all_by_frame(frame, frame_idx, not_occluded_bboxes)
        elif self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            # only export unique bounding boxes that are not occluded (first time seen)
            self.export_unique_objects(frame, frame_idx, not_occluded_bboxes)
        else:
            raise Exception("Invalid export mode")

    def append_XML_unique_object(self, filepath, object_name, object_bbox):
        object_xml = ET.SubElement(self.unique_objects_xml_tree, 'object')

        # ... image location ...
        folder_name, image_filename = os.path.split(filepath)
        filename = ET.SubElement(object_xml, 'filename')
        filename.text = image_filename
        folder = ET.SubElement(object_xml, 'folder')
        folder.text = folder_name

        name = ET.SubElement(object_xml, 'name')
        name.text = object_name

        # ensure that all boundaries are integers ...
        object_bbox = [int(x) for x in object_bbox]
        bbox_xml = ET.SubElement(object_xml, 'bbox')

        xmin = ET.SubElement(bbox_xml, 'xmin')
        xmin.text = str(object_bbox[0])  # x1
        ymin = ET.SubElement(bbox_xml, 'ymin')
        ymin.text = str(object_bbox[1])  # y1
        xmax = ET.SubElement(bbox_xml, 'xmax')
        xmax.text = str(object_bbox[2])  # x2
        ymax = ET.SubElement(bbox_xml, 'ymax')
        ymax.text = str(object_bbox[3])  # y3

    def finalize_unique_text_boxes(self):
        # save XML results
        out_xml_filename = "{0:s}/text_objects.xml".format(self.export_xml_dir)

        annotation = ET.ElementTree(self.unique_objects_xml_tree)
        annotation.write(out_xml_filename)

        # compute a single "best image" for each unique object
        for text_name in self.exported_text_objects:
            object_instances = self.exported_text_objects[text_name]

            # first, obtain object global boundaries
            all_x1, all_y1, all_x2, all_y2 = [], [], [], []
            for frame_idx, (x1, y1, x2, y2), region_img in object_instances:
                all_x1.append(x1)
                all_y1.append(y1)
                all_x2.append(x2)
                all_y2.append(y2)

            gb_x1 = min(all_x1)
            gb_y1 = min(all_y1)
            gb_x2 = max(all_x2)
            gb_y2 = max(all_y2)

            # compute average image ...
            avg_img = np.zeros((gb_y2 - gb_y1, gb_x2 - gb_x1, 3), dtype=np.float64)
            avg_count = np.zeros((gb_y2 - gb_y1, gb_x2 - gb_x1), dtype=np.int32)
            for frame_idx, (x1, y1, x2, y2), region_img in object_instances:
                off_x = x1 - gb_x1
                off_y = y1 - gb_y1
                end_y = off_y + region_img.shape[0]
                end_x = off_x + region_img.shape[1]

                avg_img[off_y:end_y, off_x:end_x] += region_img
                avg_count[off_y:end_y, off_x:end_x] += 1

            avg_mask = avg_count > 0
            avg_img[avg_mask, 0] /= avg_count[avg_mask]
            avg_img[avg_mask, 1] /= avg_count[avg_mask]
            avg_img[avg_mask, 2] /= avg_count[avg_mask]

            avg_img = avg_img.astype(np.uint8)

            # find the image with the smallest difference to the average image
            all_mse = []
            for idx, (frame_idx, (x1, y1, x2, y2), region_img) in enumerate(object_instances):
                off_x = x1 - gb_x1
                off_y = y1 - gb_y1
                end_y = off_y + region_img.shape[0]
                end_x = off_x + region_img.shape[1]

                diff = avg_img[off_y:end_y, off_x:end_x].astype(np.int32) - region_img.astype(np.int32)
                mse = np.power(diff, 2).mean()

                all_mse.append((mse, idx))

            all_mse = sorted(all_mse)

            # use the smallest difference frame ...
            final_img = object_instances[all_mse[0][1]][2]

            # export image!
            out_img_filename = "{0:s}/{1:s}.png".format(self.export_img_dir, text_name)
            # ... save image ...
            cv2.imwrite(out_img_filename, final_img)

    def finalize(self):
        if self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            self.finalize_unique_text_boxes()

    @staticmethod
    def CheckTextObject(video_object):
        # first, base decision on name prefix ..
        if video_object.id[:2].lower() == "hw":
            return True

        # other potential checks ...

        return False

    @staticmethod
    def generate_XML_objects(filepath, frame_width, frame_height, bboxes):
        annotation = ET.Element('annotation')

        # ... image size information ...
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(frame_width)
        height = ET.SubElement(size, 'height')
        height.text = str(frame_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)

        # ... image location ...
        folder_name, image_filename = os.path.split(filepath)
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_filename
        folder = ET.SubElement(annotation, 'folder')
        folder.text = folder_name

        # ... object bboxes...
        for object_name, bbox in bboxes:
            bbox = [int(x) for x in bbox]
            obj = ET.SubElement(annotation, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = 'text'
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(bbox[0]) #x1
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(bbox[1]) #y1
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(bbox[2]) #x2
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(bbox[3]) #y3
        annotation = ET.ElementTree(annotation)

        return annotation

    @staticmethod
    def FromAnnotationXML(export_mode, database, lecture, export_dir):
        # Load video annotations ....
        # ... file name ...
        annotation_suffix = database.name + "_" + lecture.title.lower()
        input_prefix = database.output_annotations + "/" + annotation_suffix
        input_main_file = input_prefix + ".xml"
        # ... get element tree object ...
        annotation_tree = ET.parse(input_main_file)
        annotation_root = annotation_tree.getroot()

        # get original rendering info ..
        draw_info_root = annotation_root.find("DrawingInfo")
        # ... canvas ...
        canvas_root = draw_info_root.find("Canvas")
        canvas_x = float(canvas_root.find("X").text)
        canvas_y = float(canvas_root.find("Y").text)
        canvas_loc = canvas_x, canvas_y
        # ... render area ...
        render_root = draw_info_root.find("Player").find("RenderArea")
        render_x = float(render_root.find("X").text)
        render_y = float(render_root.find("Y").text)
        render_w = float(render_root.find("W").text)
        render_h = float(render_root.find("H").text)
        render_loc = render_x, render_y
        render_size = render_w, render_h

        current_video_objects = []
        xml_video_objects_root = annotation_root.find('VideoObjects')
        xml_video_objects = xml_video_objects_root.findall('VideoObject')
        loading_msg = " -> Loading object: {0:s} ({1:d} Key-frames)"
        for xml_video_object in xml_video_objects:
            # load logical object ...
            video_object = VideoObject.fromXML(xml_video_object)
            print(loading_msg.format(video_object.name, len(video_object.locations)))

            current_video_objects.append(video_object)

        text_exporter = TextAnnotationExporter(export_mode, current_video_objects, canvas_loc, render_loc, render_size,
                                               export_dir)

        return text_exporter


