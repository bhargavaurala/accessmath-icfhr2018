
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.preprocessing.content.labeler import Labeler
from .keyframe_portion import KeyFramePortion
from .video_object import VideoObject


class KeyFrameAnnotation:
    def __init__(self, database, lecture, frame_idx, frame_time, frame_objects, raw_image):
        self.database = database
        self.lecture = lecture
        self.idx = frame_idx
        self.time = frame_time
        self.objects = frame_objects

        self.portions = []

        self.raw_image = raw_image

        if raw_image is not None:
            gray_scale = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2GRAY)
            self.grayscale_image = np.zeros(raw_image.shape, dtype=np.uint8)
            self.grayscale_image[:, :, 0] = gray_scale
            self.grayscale_image[:, :, 1] = gray_scale
            self.grayscale_image[:, :, 2] = gray_scale

        else:
            self.grayscale_image = None

        self.object_mask = None
        if raw_image is not None:
            self.update_object_mask()

        self.binary_image = None
        self.binary_cc = None
        self.combined_image = None

        if raw_image is not None:
            self.update_binary_image(False)


    def __repr__(self):
        lect_str = self.database + " - " + self.lecture
        loc_str = str(self.idx) + " at " + TimeHelper.stampToStr(self.time)
        return "{Keyframe: [" + lect_str + "], [" + loc_str + "]}\n"

    def ccs_in_region(self, min_x, max_x, min_y, max_y):
        if self.binary_cc is None:
            return []
        else:
            in_region = []
            for cc in self.binary_cc:
                if min_x <= cc.min_x and cc.max_x <= max_x and min_y <= cc.min_y and cc.max_y <= max_y:
                    in_region.append(cc)

            return in_region

    def get_CCs_by_ID(self):
        if self.binary_cc is None:
            return {}
        else:
            ccs_by_id = {}

            for cc in self.binary_cc:
                ccs_by_id[cc.strID()] = cc

            return ccs_by_id

    def check_cc_overlaps_background(self, cc):
        assert isinstance(cc, ConnectedComponent)

        if cc.max_x < 0 or cc.min_x >= self.object_mask.shape[1] or cc.max_y < 0 or cc.min_y >= self.object_mask.shape[0]:
            # completely out of boundaries and assumed as part of the background ...
            return True

        # use pre-computed object mask
        # first, find the cut for the given cc
        mask_cut = self.object_mask[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]

        # in case that the given CC has a bounding box out of boundaries of this key-frame
        cc_start_x = max(0, -cc.min_x)
        cc_start_y = max(0, -cc.min_y)
        cc_cut = cc.img[cc_start_y:cc_start_y + mask_cut.shape[0], cc_start_x:cc_start_x + mask_cut.shape[1]]

        pixels_overlap = np.count_nonzero(np.logical_and(cc_cut, mask_cut))
        # print((cc.min_x, cc.max_x, cc.min_y, cc.max_y))
        # print(mask_cut.shape)

        return pixels_overlap > 0

    def get_XML_string(self):
        xml_string = "<KeyFrame>\n"
        xml_string += "  <Index>" + str(self.idx) + "</Index>\n"
        xml_string += "  <Portions>\n"
        for portion in self.portions:
            xml_string += portion.get_XML_string()
        xml_string += "  </Portions>\n"
        xml_string += "</KeyFrame>\n"

        return xml_string

    def add_portion(self, new_portion):
        self.portions.append(new_portion)
        self.update_binary_image(True)

    def del_portion(self, portion_idx):
        if 0 <= portion_idx < len(self.portions):
            del self.portions[portion_idx]
            self.update_binary_image(True)

    def update_object_mask(self):
        h, w, _ = self.raw_image.shape
        self.object_mask = np.zeros((h, w), dtype=np.bool)

        for object in self.objects:
            loc = object.locations[0]
            start_y = max(int(loc.y), 0)
            end_y = min(int(loc.y + loc.h), h - 1)
            start_x = max(int(loc.x), 0)
            end_x = min(int(loc.x + loc.w), w - 1)

            self.object_mask[start_y:end_y, start_x:end_x] = True

    def update_binary_cc(self, verbose=True):
        h, w, _ = self.binary_image.shape
        if verbose:
            print("Computing CC for frame: " + str(self.idx))

        fake_age = np.zeros((h, w), dtype=np.float32)
        current_cc = Labeler.extractSpatioTemporalContent((255 - self.binary_image[:, :, 0]), fake_age, False)
        self.binary_cc = current_cc

        if verbose:
            print("    Found: " + str(len(current_cc)) + " CCs")


    def update_combined_image(self):
        if self.raw_image is None:
            return

        # now, create the combined image as well ...
        self.combined_image = self.grayscale_image.copy()

        # update combined
        inverse_binary_mask = np.logical_not(self.binary_image[:, :, 0])
        self.combined_image[inverse_binary_mask, 2] = 255
        self.combined_image[self.object_mask, 0] = 255

    def update_binary_image(self, update_cc=False):
        h, w, _ = self.raw_image.shape

        # first, merge all blocks (portions) into a single binary image
        # (using OR operation)

        # start with empty white image by default ...
        self.binary_image = np.ones((h, w, 3), dtype=np.uint8) * 255

        if len(self.portions) > 0:
            # update from portions ...
            tempo_binary = np.zeros((h, w), dtype=np.int32)
            for portion in self.portions:
                # tempo_binary[portion.y:portion.y + portion.h, portion.x:portion.x + portion.w] = portion.binary.copy()
                # combine the inverse of the binary (make writing white, background black)
                tempo_binary[portion.y:portion.y + portion.h, portion.x:portion.x + portion.w] += (255 - portion.binary)

            # anything that is not black (background) is normalized to the maximum (writing == 255)
            tempo_binary[tempo_binary > 0] = 255
            # invert the result (writing = 0 [black], background = 255 [white])
            tempo_binary = 255 - tempo_binary

            self.binary_image[:, :, 0] = tempo_binary.copy()
            self.binary_image[:, :, 1] = tempo_binary.copy()
            self.binary_image[:, :, 2] = tempo_binary.copy()

        self.update_combined_image()

        if update_cc:
            self.update_binary_cc()

    def save(self, file_prefix):
        # this step should also export the binary image
        pass

    @staticmethod
    def LoadExportedKeyframes(xml_filename, image_prefix, load_segments=False, swap_red_blue=True, binary_mode=False):
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        namespace = VideoObject.XMLNamespace
        database_name = root.find(namespace + 'Database').text
        lecture_name = root.find(namespace + 'Lecture').text

        keyframes_root = root.find(namespace + 'VideoKeyFrames')
        keyframes_xml_roots = keyframes_root.findall(namespace + 'VideoKeyFrame')

        extracted_keyframes = []
        object_ids = {}
        next_object_id = 1
        for xml_keyframe in keyframes_xml_roots:
            # Load key-frame metadata
            frame_idx = int(xml_keyframe.find(namespace + "Index").text)
            frame_time = float(xml_keyframe.find(namespace + "AbsTime").text)

            video_objects_root = xml_keyframe.find(namespace + 'VideoObjects')
            video_objects_xml_roots = video_objects_root.findall(namespace + "VideoObject")

            frame_objects = []
            for xml_video_object in video_objects_xml_roots:
                object_name = xml_video_object.find(namespace + 'Name').text
                loc_x = float(xml_video_object.find(namespace + 'X').text)
                loc_y = float(xml_video_object.find(namespace + 'Y').text)
                loc_w = float(xml_video_object.find(namespace + 'W').text)
                loc_h = float(xml_video_object.find(namespace + 'H').text)

                if object_name not in object_ids:
                    object_ids[object_name] = next_object_id
                    next_object_id += 1

                current_id = object_ids[object_name]

                video_object = VideoObject(current_id, object_name)
                video_object.set_location_at(frame_idx, frame_time, True, loc_x, loc_y, loc_w, loc_h)

                frame_objects.append(video_object)

            # Load key-frame image
            frame_img = cv2.imread(image_prefix + str(frame_idx) + ".png")

            if not binary_mode:
                # switch color channels ...
                if swap_red_blue:
                    tempo_channel = frame_img[:, :, 0].copy()
                    frame_img[:, :, 0] = frame_img[:, :, 2]
                    frame_img[:, :, 2] = tempo_channel

                # created keyframe object
                keyframe = KeyFrameAnnotation(database_name, lecture_name, frame_idx, frame_time, frame_objects, frame_img)
            else:
                # assume that provided image is binary ...
                keyframe = KeyFrameAnnotation(database_name, lecture_name, frame_idx, frame_time, frame_objects, None)
                keyframe.binary_image = frame_img

            extracted_keyframes.append(keyframe)

        if load_segments:
            segments_root = root.find(namespace + 'VideoSegments')
            segments_xml_roots = segments_root.findall(namespace + 'VideoSegment')

            segments = []
            for xml_segment in segments_xml_roots:
                segment_start = int(xml_segment.find(namespace + "Start").text)
                segment_end = int(xml_segment.find(namespace + "End").text)

                segments.append((segment_start, segment_end))

            return extracted_keyframes, segments
        else:
            return extracted_keyframes

    def __getitem__(self, item):
        return self.portions[item]

    @staticmethod
    def LoadKeyframesPortions(xml_filename, keyframes, portions_path):
        tempo_kf_index = {kf.idx: kf for kf in keyframes}

        tree = ET.parse(xml_filename)
        root = tree.getroot()

        namespace = VideoObject.XMLNamespace

        keyframes_root = root.find(namespace + 'KeyFrames')
        keyframes_xml_roots = keyframes_root.findall(namespace + 'KeyFrame')

        for xml_keyframe in keyframes_xml_roots:
            kf_idx = int(xml_keyframe.find(namespace + 'Index').text)
            if kf_idx in tempo_kf_index:
                portions_root = xml_keyframe.find(namespace + 'Portions')
                portions_xml_roots = portions_root.findall(namespace + 'KeyFramePortion')

                tempo_portions = []
                for idx, xml_portion in enumerate(portions_xml_roots):
                    # load portion binary ...
                    binary = cv2.imread(portions_path + "/frame_" + str(kf_idx) + "/" + str(idx) + ".png")

                    px = int(xml_portion.find(namespace + 'X').text)
                    py = int(xml_portion.find(namespace + 'Y').text)
                    pw = int(xml_portion.find(namespace + 'W').text)
                    ph = int(xml_portion.find(namespace + 'H').text)

                    if py + ph > tempo_kf_index[kf_idx].raw_image.shape[0]:
                        print("*** Portion out of boundary for height = " + str(ph) + "! (" + str(py + ph) + ")")
                        ph = tempo_kf_index[kf_idx].raw_image.shape[0] - py

                        print("*** New Height: " + str(ph))

                    if px + pw > tempo_kf_index[kf_idx].raw_image.shape[1]:
                        print("*** Portion out of boundary for width = " + str(pw) + "! (" + str(px + pw) + ")")
                        pw = tempo_kf_index[kf_idx].raw_image.shape[1] - px

                        print("*** New Width: " + str(pw))

                    tempo_portions.append(KeyFramePortion(px, py, pw, ph, binary[:, :, 0]))

                tempo_kf_index[kf_idx].portions = tempo_portions
                tempo_kf_index[kf_idx].update_binary_image(True)
            else:
                print("Unknown Key-frame found in annotations: " + str(kf_idx))

    @staticmethod
    def CombineKeyframesPerSegment(keyframes, segments, use_portions):
        segment_keyframes = [[] for segment in segments]

        # split key-frames by segment
        last_segment = 0
        for keyframe in keyframes:
            # Assume that segments are ordered ...
            while keyframe.idx > segments[last_segment][1]:
                last_segment += 1

            # add ...
            segment_keyframes[last_segment].append(keyframe)

        # combine by segment
        combined_keyframes = []
        for keyframe_list in segment_keyframes:
            if len(keyframe_list) == 1:
                # exactly one key-frame on segment, no need to combine
                combined_keyframes.append(keyframe_list[0])
            else:
                # use information from the last frame
                last = keyframe_list[-1]

                # find common objects ...
                object_instances = {}
                for keyframe in keyframe_list:
                    for object in keyframe.objects:
                        # object_instances
                        if not object.id in object_instances:
                            object_instances[object.id] = [object]
                        else:
                            object_instances[object.id].append(object)

                comb_objects = []
                for object_id in object_instances:
                    if len(object_instances[object_id]) == len(keyframe_list):
                        locs = [instance.locations[0] for instance in  object_instances[object_id]]
                        all_x, all_y, all_r, all_b = zip(*[(loc.x, loc.y, loc.x + loc.w, loc.y + loc.h) for loc in locs])
                        comb_x = max(all_x)
                        comb_y = max(all_y)
                        comb_w = min(all_r) - comb_x
                        comb_h = min(all_b) - comb_y

                        first_instance = object_instances[object_id][0]
                        comb_object = VideoObject(first_instance.id, first_instance.name)
                        comb_object.set_location_at(last.idx, last.time, True, comb_x, comb_y, comb_w, comb_h)

                        comb_objects.append(comb_object)

                new_keyframe = KeyFrameAnnotation(last.database, last.lecture, last.idx, last.time, comb_objects,
                                                  last.raw_image)

                if use_portions:
                    # combine portions from all key-frames to create binary representation ...
                    combined_portions = []
                    for keyframe in keyframe_list:
                        combined_portions += keyframe.portions

                    # merge from all portions, and recompute binary and CCs
                    new_keyframe.portions = combined_portions
                    new_keyframe.update_binary_image(True)
                else:
                    # create combined key-frame ...
                    combined_binary = np.ones(keyframes[0].binary_image.shape, dtype=np.uint8) * 255

                    # first the binary image (use And because background is assumed white)
                    for keyframe in keyframe_list:
                        combined_binary = np.logical_and(combined_binary, keyframe.binary_image).astype(np.uint8) * 255

                    # use pre-computed binary image, and update CCs only
                    new_keyframe.binary_image = combined_binary
                    new_keyframe.update_binary_cc()

                # new_keyframe.update_object_mask()
                new_keyframe.update_combined_image()

                combined_keyframes.append(new_keyframe)

        return combined_keyframes


