import os

import cv2
import numpy as np

from AM_CommonTools.data.connected_component import ConnectedComponent
from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.unique_cc_group import UniqueCCGroup
from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_canvas import ScreenCanvas
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AccessMath.interface.controls.screen_image import ScreenImage
from AccessMath.interface.controls.screen_label import ScreenLabel
from AccessMath.util.visualizer import Visualizer


class GTUniqueCCAnnotator(Screen):
    ModeNavigate = 0
    ModeMatch_RegionSelection = 1
    ModeMatch_Matching = 2
    ModeMatch_Remove = 3
    ModeExitConfirm = 4

    ViewModeRaw = 0
    ViewModeGray = 1
    ViewModeBinary = 2
    ViewModeColored = 3

    CCShowColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255),
                    (128, 0, 0), (0, 128, 0), (0, 0, 128),
                    (128, 128, 0), (128, 0, 128), (0, 128, 128),
                    (255, 128, 0), (255, 0, 128), (0, 255, 128),
                    (128, 255, 0), (128, 0, 255), (0, 128, 255)]

    ParamsMinRecall = 10
    ParamsMinPrecision = 10
    ParamsMaxTranslation = 10

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Unique CC Ground Truth Annotation Interface", size)

        general_background = (100, 90, 80)
        text_color = (255, 255, 255)
        button_text_color = (35, 50, 20)
        button_back_color = (228, 228, 228)
        self.elements.back_color = general_background

        self.db_name = db_name
        self.lecture_title = lecture_title

        self.output_path = output_path

        export_filename = self.output_path + "/segments.xml"
        export_image_prefix = self.output_path + "/keyframes/"
        # load including segment information
        self.keyframe_annotations, self.segments = KeyFrameAnnotation.LoadExportedKeyframes(export_filename, export_image_prefix, True)

        if len(self.keyframe_annotations) > 0:
            print("Key-frames Loaded: " + str(len(self.keyframe_annotations)))
        else:
            raise Exception("Cannot start with 0 key-frames")

        portions_filename = self.output_path + "/portions.xml"
        portions_path = self.output_path + "/portions/"
        if os.path.exists(portions_filename):
            # Saved data detected, loading
            print("Previously saved portion data detected, loading")
            KeyFrameAnnotation.LoadKeyframesPortions(portions_filename, self.keyframe_annotations, portions_path)
        else:
            raise Exception("No saved portion data detected, cannot continue")

        print("Original Key-frames: " + str(len(self.keyframe_annotations)))
        print("Segments: " + str(len(self.segments)))

        self.keyframe_annotations = KeyFrameAnnotation.CombineKeyframesPerSegment(self.keyframe_annotations, self.segments, True)
        print("Key-frames after combination per segment" + str(len(self.keyframe_annotations)))

        # other CC/group elements
        self.unique_groups = None
        self.cc_group = None
        self.colored_cache = []
        self.cc_total = 0
        for kf_idx, keyframe in enumerate(self.keyframe_annotations):
            self.cc_total += len(keyframe.binary_cc)

        unique_cc_filename = self.output_path + "/unique_ccs.xml"
        if os.path.exists(unique_cc_filename):
            # Saved data detected, loading
            print("Previously saved unique CC data detected, loading")

            self.cc_group, self.unique_groups = UniqueCCGroup.GroupsFromXML(self.keyframe_annotations, unique_cc_filename)
        else:
            # no previous data, build default index (all CCs are unique)
            self.unique_groups = []
            self.cc_group = []
            for kf_idx, keyframe in enumerate(self.keyframe_annotations):
                self.cc_group.append({})

                for cc in keyframe.binary_cc:
                    new_group = UniqueCCGroup(cc, kf_idx)
                    self.unique_groups.append(new_group)
                    self.cc_group[kf_idx][cc.strID()] = new_group

        self.update_colored_cache(0)

        self.view_mode = GTUniqueCCAnnotator.ViewModeColored
        self.edition_mode = GTUniqueCCAnnotator.ModeNavigate
        self.view_scale = 1.0
        self.selected_keyframe = 0

        self.matching_delta_x = 0
        self.matching_delta_y = 0
        self.matching_scores = None
        self.matching_min_recall = 0.99
        self.matching_min_precision = 0.99
        self.base_matching = None

        # add elements....
        container_top = 10
        container_width = 330

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        # Navigation panel to move accross frames
        self.container_nav_buttons = ScreenContainer("container_nav_buttons", (container_width, 70), back_color=general_background)
        self.container_nav_buttons.position = (self.width - self.container_nav_buttons.width - 10, container_top)
        self.elements.append(self.container_nav_buttons)

        self.lbl_nav_keyframe = ScreenLabel("lbl_nav_keyframe", "Key-Frame: 1 / " + str(len(self.keyframe_annotations)), 21, 290, 1)
        self.lbl_nav_keyframe.position = (5, 5)
        self.lbl_nav_keyframe.set_background(general_background)
        self.lbl_nav_keyframe.set_color(text_color)
        self.container_nav_buttons.append(self.lbl_nav_keyframe)

        time_str = TimeHelper.stampToStr(self.keyframe_annotations[self.selected_keyframe].time)
        self.lbl_nav_time = ScreenLabel("lbl_nav_time", time_str, 21, 290, 1)
        self.lbl_nav_time.position = (5, self.lbl_nav_keyframe.get_bottom() + 20)
        self.lbl_nav_time.set_background(general_background)
        self.lbl_nav_time.set_color(text_color)
        self.container_nav_buttons.append(self.lbl_nav_time)

        self.btn_nav_keyframe_prev = ScreenButton("btn_nav_keyframe_prev", "Prev", 21, 90)
        self.btn_nav_keyframe_prev.set_colors(button_text_color, button_back_color)
        self.btn_nav_keyframe_prev.position = (10, self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_nav_keyframe_prev.click_callback = self.btn_nav_keyframe_prev_click
        self.container_nav_buttons.append(self.btn_nav_keyframe_prev)

        self.btn_nav_keyframe_next = ScreenButton("btn_nav_keyframe_next", "Next", 21, 90)
        self.btn_nav_keyframe_next.set_colors(button_text_color, button_back_color)
        self.btn_nav_keyframe_next.position = (self.container_nav_buttons.width - self.btn_nav_keyframe_next.width - 10,
                                               self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_nav_keyframe_next.click_callback = self.btn_nav_keyframe_next_click
        self.container_nav_buttons.append(self.btn_nav_keyframe_next)

        # confirmation panel
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (container_width, 70), back_color=general_background)
        self.container_confirm_buttons.position = (self.width - self.container_confirm_buttons.width - 10, container_top)
        self.elements.append(self.container_confirm_buttons)
        self.container_confirm_buttons.visible = False

        self.lbl_confirm_message = ScreenLabel("lbl_confirm_message", "Confirmation message goes here?", 21, 290, 1)
        self.lbl_confirm_message.position = (5, 5)
        self.lbl_confirm_message.set_background(general_background)
        self.lbl_confirm_message.set_color(text_color)
        self.container_confirm_buttons.append(self.lbl_confirm_message)

        self.btn_confirm_cancel = ScreenButton("btn_confirm_cancel", "Cancel", 21, 130)
        self.btn_confirm_cancel.set_colors(button_text_color, button_back_color)
        self.btn_confirm_cancel.position = (10, self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_confirm_cancel.click_callback = self.btn_confirm_cancel_click
        self.container_confirm_buttons.append(self.btn_confirm_cancel)

        self.btn_confirm_accept = ScreenButton("btn_confirm_accept", "Accept", 21, 130)
        self.btn_confirm_accept.set_colors(button_text_color, button_back_color)
        self.btn_confirm_accept.position = (self.container_confirm_buttons.width - self.btn_confirm_accept.width - 10,
                                            self.lbl_confirm_message.get_bottom() + 10)
        self.btn_confirm_accept.click_callback = self.btn_confirm_accept_click
        self.container_confirm_buttons.append(self.btn_confirm_accept)

        # View panel with view control buttons
        self.container_view_buttons = ScreenContainer("container_view_buttons", (container_width, 165), back_color=general_background)
        self.container_view_buttons.position = (self.width - self.container_view_buttons.width - 10,
                                                self.container_nav_buttons.get_bottom() + 10)
        self.elements.append(self.container_view_buttons)


        button_width = 190
        button_left = (self.container_view_buttons.width - button_width) / 2

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", 21, container_width - 10, 1)
        self.lbl_zoom.position = (5, 5)
        self.lbl_zoom.set_background(general_background)
        self.lbl_zoom.set_color(text_color)
        self.container_view_buttons.append(self.lbl_zoom)

        self.btn_zoom_reduce = ScreenButton("btn_zoom_reduce", "[ - ]", 21, 90)
        self.btn_zoom_reduce.set_colors(button_text_color, button_back_color)
        self.btn_zoom_reduce.position = (10, self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_reduce.click_callback = self.btn_zoom_reduce_click
        self.container_view_buttons.append(self.btn_zoom_reduce)

        self.btn_zoom_increase = ScreenButton("btn_zoom_increase", "[ + ]", 21, 90)
        self.btn_zoom_increase.set_colors(button_text_color, button_back_color)
        self.btn_zoom_increase.position = (self.container_view_buttons.width - self.btn_zoom_increase.width - 10,
                                           self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_increase.click_callback = self.btn_zoom_increase_click
        self.container_view_buttons.append(self.btn_zoom_increase)

        self.btn_zoom_zero = ScreenButton("btn_zoom_zero", "100%", 21, 90)
        self.btn_zoom_zero.set_colors(button_text_color, button_back_color)
        self.btn_zoom_zero.position = ((self.container_view_buttons.width - self.btn_zoom_zero.width) / 2,
                                       self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_zero.click_callback = self.btn_zoom_zero_click
        self.container_view_buttons.append(self.btn_zoom_zero)

        self.btn_view_raw = ScreenButton("btn_view_raw", "Raw View", 21, button_2_width)
        self.btn_view_raw.set_colors(button_text_color, button_back_color)
        self.btn_view_raw.position = (button_2_left, self.btn_zoom_zero.get_bottom() + 10)
        self.btn_view_raw.click_callback = self.btn_view_raw_click
        self.container_view_buttons.append(self.btn_view_raw)

        self.btn_view_gray = ScreenButton("btn_view_gray", "Grayscale View", 21, button_2_width)
        self.btn_view_gray.set_colors(button_text_color, button_back_color)
        self.btn_view_gray.position = (button_2_right, self.btn_zoom_zero.get_bottom() + 10)
        self.btn_view_gray.click_callback = self.btn_view_gray_click
        self.container_view_buttons.append(self.btn_view_gray)

        self.btn_view_binary = ScreenButton("btn_view_binary", "Binary View", 21, button_2_width)
        self.btn_view_binary.set_colors(button_text_color, button_back_color)
        self.btn_view_binary.position = (button_2_left, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_binary.click_callback = self.btn_view_binary_click
        self.container_view_buttons.append(self.btn_view_binary)

        self.btn_view_colored = ScreenButton("btn_view_colored", "Colored View", 21, button_2_width)
        self.btn_view_colored.set_colors(button_text_color, button_back_color)
        self.btn_view_colored.position = (button_2_right, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_colored.click_callback = self.btn_view_colored_click
        self.container_view_buttons.append(self.btn_view_colored)

        # Panel with action buttons (Add/Remove links)
        self.container_action_buttons = ScreenContainer("container_action_buttons", (container_width, 45),
                                                        general_background)
        self.container_action_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_action_buttons)

        self.btn_matches_add = ScreenButton("btn_matches_add", "Add Matches", 21, button_2_width)
        self.btn_matches_add.set_colors(button_text_color, button_back_color)
        self.btn_matches_add.position = (button_2_left, 5)
        self.btn_matches_add.click_callback = self.btn_matches_add_click
        self.container_action_buttons.append(self.btn_matches_add)

        self.btn_matches_del = ScreenButton("btn_matches_del", "Del. Matches", 21, button_2_width)
        self.btn_matches_del.set_colors(button_text_color, button_back_color)
        self.btn_matches_del.position = (button_2_right, 5)
        self.btn_matches_del.click_callback = self.btn_matches_del_click
        self.container_action_buttons.append(self.btn_matches_del)

        # ===============================================
        # Panel with matching parameters for step 1 (Matching Translation)
        self.container_matching_translation = ScreenContainer("container_matching_translation", (container_width, 150), general_background)
        self.container_matching_translation.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_matching_translation)

        self.lbl_translation_title = ScreenLabel("lbl_translation_title", "Translation Parameters", 21, container_width - 10, 1)
        self.lbl_translation_title.position = (5, 5)
        self.lbl_translation_title.set_background(general_background)
        self.lbl_translation_title.set_color(text_color)
        self.container_matching_translation.append(self.lbl_translation_title)

        self.lbl_delta_x = ScreenLabel("lbl_delta_x", "Delta X: " + str(self.matching_delta_x), 21, container_width - 10, 1)
        self.lbl_delta_x.position = (5, self.lbl_translation_title.get_bottom() + 20)
        self.lbl_delta_x.set_background(general_background)
        self.lbl_delta_x.set_color(text_color)
        self.container_matching_translation.append(self.lbl_delta_x)

        max_delta = GTUniqueCCAnnotator.ParamsMaxTranslation
        self.scroll_delta_x = ScreenHorizontalScroll("scroll_delta_x", -max_delta, max_delta, 0, 1)
        self.scroll_delta_x.position = (5, self.lbl_delta_x.get_bottom() + 10)
        self.scroll_delta_x.width = container_width - 10
        self.scroll_delta_x.scroll_callback = self.scroll_delta_x_change
        self.container_matching_translation.append(self.scroll_delta_x)

        self.lbl_delta_y = ScreenLabel("lbl_delta_y", "Delta Y: " + str(self.matching_delta_y), 21, container_width - 10, 1)
        self.lbl_delta_y.position = (5, self.scroll_delta_x.get_bottom() + 20)
        self.lbl_delta_y.set_background(general_background)
        self.lbl_delta_y.set_color(text_color)
        self.container_matching_translation.append(self.lbl_delta_y)

        self.scroll_delta_y = ScreenHorizontalScroll("scroll_delta_y", -max_delta, max_delta, 0, 1)
        self.scroll_delta_y.position = (5, self.lbl_delta_y.get_bottom() + 10)
        self.scroll_delta_y.width = container_width - 10
        self.scroll_delta_y.scroll_callback = self.scroll_delta_y_change
        self.container_matching_translation.append(self.scroll_delta_y)

        self.container_matching_translation.visible = False

        # ===============================================
        # Panel with matching parameters for step 2 (Matching Strictness)
        self.container_matching_strictness = ScreenContainer("container_matching_strictness", (container_width, 150),
                                                              general_background)
        self.container_matching_strictness.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_matching_strictness)

        self.lbl_matching_title = ScreenLabel("lbl_matching_title", "Matching Parameters", 21,
                                                 container_width - 10, 1)
        self.lbl_matching_title.position = (5, 5)
        self.lbl_matching_title.set_background(general_background)
        self.lbl_matching_title.set_color(text_color)
        self.container_matching_strictness.append(self.lbl_matching_title)

        str_recall = "Minimum Recall: " + str(int(self.matching_min_recall * 100))
        self.lbl_min_recall = ScreenLabel("lbl_min_recall", str_recall, 21, container_width - 10, 1)
        self.lbl_min_recall.position = (5, self.lbl_matching_title.get_bottom() + 20)
        self.lbl_min_recall.set_background(general_background)
        self.lbl_min_recall.set_color(text_color)
        self.container_matching_strictness.append(self.lbl_min_recall)

        min_recall = GTUniqueCCAnnotator.ParamsMinRecall
        self.scroll_min_recall = ScreenHorizontalScroll("scroll_min_recall", min_recall, 100, 99, 1)
        self.scroll_min_recall.position = (5, self.lbl_min_recall.get_bottom() + 10)
        self.scroll_min_recall.width = container_width - 10
        self.scroll_min_recall.scroll_callback = self.scroll_min_recall_change
        self.container_matching_strictness.append(self.scroll_min_recall)

        str_precision = "Minimum Precision: " + str(int(self.matching_min_precision * 100))
        self.lbl_min_precision = ScreenLabel("lbl_min_precision", str_precision, 21, container_width - 10, 1)
        self.lbl_min_precision.position = (5, self.scroll_min_recall.get_bottom() + 20)
        self.lbl_min_precision.set_background(general_background)
        self.lbl_min_precision.set_color(text_color)
        self.container_matching_strictness.append(self.lbl_min_precision)

        min_precision = GTUniqueCCAnnotator.ParamsMinPrecision
        self.scroll_min_precision = ScreenHorizontalScroll("scroll_min_precision", min_precision, 100, 99, 1)
        self.scroll_min_precision.position = (5, self.lbl_min_precision.get_bottom() + 10)
        self.scroll_min_precision.width = container_width - 10
        self.scroll_min_precision.scroll_callback = self.scroll_min_precision_change
        self.container_matching_strictness.append(self.scroll_min_precision)

        self.container_matching_strictness.visible = False

        # ===============================================
        stats_background = (60, 50, 40)
        self.container_stats = ScreenContainer("container_stats", (container_width, 70), back_color=stats_background)
        self.container_stats.position = (self.width - container_width - 10, self.container_action_buttons.get_bottom() + 5)
        self.elements.append(self.container_stats)

        self.lbl_cc_stats = ScreenLabel("lbl_cc_stats", "Connected Component Stats", 21, container_width - 10, 1)
        self.lbl_cc_stats.position = (5, 5)
        self.lbl_cc_stats.set_background(stats_background)
        self.lbl_cc_stats.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats)

        self.lbl_cc_raw = ScreenLabel("lbl_cc_raw", "Total Raw CC:\n" + str(self.cc_total), 21, button_2_width, 1)
        self.lbl_cc_raw.position = (button_2_left, self.lbl_cc_stats.get_bottom() + 10)
        self.lbl_cc_raw.set_background(stats_background)
        self.lbl_cc_raw.set_color(text_color)
        self.container_stats.append(self.lbl_cc_raw)

        self.lbl_cc_unique = ScreenLabel("lbl_cc_unique", "Total Unique CC:\n" + str(len(self.unique_groups)), 21, button_2_width, 1)
        self.lbl_cc_unique.position = (button_2_right, self.lbl_cc_stats.get_bottom() + 10)
        self.lbl_cc_unique.set_background(stats_background)
        self.lbl_cc_unique.set_color(text_color)
        self.container_stats.append(self.lbl_cc_unique)

        #=============================================================
        # Panel with state buttons (Undo, Redo, Save)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 200),
                                                       general_background)
        self.container_state_buttons.position = (
        self.container_view_buttons.get_left(), self.container_stats.get_bottom() + 10)
        self.elements.append(self.container_state_buttons)

        self.btn_undo = ScreenButton("btn_undo", "Undo", 21, button_width)
        self.btn_undo.set_colors(button_text_color, button_back_color)
        self.btn_undo.position = (button_left, 5)
        self.btn_undo.click_callback = self.btn_undo_click
        self.container_state_buttons.append(self.btn_undo)

        self.btn_redo = ScreenButton("btn_redo", "Redo", 21, button_width)
        self.btn_redo.set_colors(button_text_color, button_back_color)
        self.btn_redo.position = (button_left, self.btn_undo.get_bottom() + 10)
        self.btn_redo.click_callback = self.btn_redo_click
        self.container_state_buttons.append(self.btn_redo)

        self.btn_save = ScreenButton("btn_save", "Save", 21, button_width)
        self.btn_save.set_colors(button_text_color, button_back_color)
        self.btn_save.position = (button_left, self.btn_redo.get_bottom() + 10)
        self.btn_save.click_callback = self.btn_save_click
        self.container_state_buttons.append(self.btn_save)

        self.btn_exit = ScreenButton("btn_exit", "Exit", 21, button_width)
        self.btn_exit.set_colors(button_text_color, button_back_color)
        self.btn_exit.position = (button_left, self.btn_save.get_bottom() + 30)
        self.btn_exit.click_callback = self.btn_exit_click
        self.container_state_buttons.append(self.btn_exit)

        # print("MAKE CONTAINER STATS VISIBLE AGAIN!!!")
        # self.container_stats.visible = False
        # self.container_state_buttons.visible = False

        # ==============================================================
        image_width = self.width - self.container_nav_buttons.width - 30
        image_height = self.height - container_top - 10
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, container_top)
        self.elements.append(self.container_images)

        # ... image objects ...
        tempo_blank = np.zeros((50, 50, 3), np.uint8)
        tempo_blank[:, :, :] = 255
        self.img_main = ScreenImage("img_main", tempo_blank, 0, 0, True, cv2.INTER_NEAREST)
        self.img_main.position = (0, 0)
        #self.img_main.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_main)

        # canvas used for annotations
        self.canvas_select = ScreenCanvas("canvas_select", 100, 100)
        self.canvas_select.position = (0, 0)
        self.canvas_select.locked = False
        # self.canvas_select.object_edited_callback = self.canvas_object_edited
        # self.canvas_select.object_selected_callback = self.canvas_selection_changed
        self.container_images.append(self.canvas_select)

        self.canvas_select.add_element("selection_rectangle", 10, 10, 40, 40)
        self.canvas_select.elements["selection_rectangle"].visible = False

        self.undo_stack = []
        self.redo_stack = []

        self.update_current_view(True)

    def update_colored_cache(self, start_frame):
        # remove outdated key-frames
        while len(self.colored_cache) > start_frame:
            del self.colored_cache[start_frame]

        # compute new frames ...
        n_colors = len(GTUniqueCCAnnotator.CCShowColors)
        for kf_idx in range(start_frame, len(self.keyframe_annotations)):
            keyframe = self.keyframe_annotations[kf_idx]
            colored_image = np.zeros(keyframe.binary_image.shape, dtype=np.uint8)

            for idx, cc in enumerate(keyframe.binary_cc):
                start = self.cc_group[kf_idx][cc.strID()].start_frame
                cc_color = GTUniqueCCAnnotator.CCShowColors[start % n_colors]

                current_cut = colored_image[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                cc_mask = cc.img > 0
                current_cut[cc_mask, 0] += cc_color[0]
                current_cut[cc_mask, 1] += cc_color[1]
                current_cut[cc_mask, 2] += cc_color[2]

            background_mask = colored_image.sum(axis=2) == 0
            colored_image[background_mask, 0] = 96
            colored_image[background_mask, 1] = 96
            colored_image[background_mask, 2] = 96

            # cv2.imshow("tempo", colored_image)
            # cv2.waitKey()
            self.colored_cache.append(colored_image)

    def update_current_view(self, resized=False):
        if (self.edition_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection or
            self.edition_mode == GTUniqueCCAnnotator.ModeMatch_Matching):
            # special override case, override current view to show matching image
            base_image = self.base_matching
        elif self.view_mode == GTUniqueCCAnnotator.ViewModeGray:
            base_image = self.keyframe_annotations[self.selected_keyframe].grayscale_image
        elif self.view_mode == GTUniqueCCAnnotator.ViewModeBinary:
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        elif self.view_mode == GTUniqueCCAnnotator.ViewModeColored:
            base_image = self.colored_cache[self.selected_keyframe]
        else:
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image

        h, w, c = base_image.shape
        modified_image = base_image.copy()

        if self.edition_mode == GTUniqueCCAnnotator.ModeMatch_Matching:
            # color the affected CC by the current selection
            for precision, recall, prev_cc, curr_cc in self.matching_scores:
                current_cut = modified_image[curr_cc.min_y:curr_cc.max_y + 1, curr_cc.min_x:curr_cc.max_x + 1]
                cc_mask = curr_cc.img > 0

                if precision >= self.matching_min_precision and recall >= self.matching_min_recall:
                    # show as a match
                    current_cut[cc_mask, 0] = 0
                    current_cut[cc_mask, 1] = 255
                    current_cut[cc_mask, 2] = 0
                else:
                    # show as a mismatch
                    current_cut[cc_mask, 0] = 255
                    current_cut[cc_mask, 1] = 0
                    current_cut[cc_mask, 2] = 0


        # finally, resize ...
        modified_image = cv2.resize(modified_image, (int(w * self.view_scale), int(h * self.view_scale)),
                                    interpolation=cv2.INTER_NEAREST)

        self.canvas_select.height, self.canvas_select.width, _ = modified_image.shape

        # replace/update image
        self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        if resized:
            self.container_images.recalculate_size()

    def update_selected_keyframe(self, new_selected):
        if 0 <= new_selected < len(self.keyframe_annotations):
            self.selected_keyframe = new_selected
        else:
            return

        self.lbl_nav_keyframe.set_text("Key-Frame: " + str(self.selected_keyframe + 1) + " / " +
                                       str(len(self.keyframe_annotations)))

        time_str = TimeHelper.stampToStr(self.keyframe_annotations[self.selected_keyframe].time)
        self.lbl_nav_time.set_text(time_str)

        self.update_current_view()

    def btn_nav_keyframe_next_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe + 1)

    def btn_nav_keyframe_prev_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe - 1)

    def update_view_scale(self, new_scale):
        prev_scale = self.view_scale

        if 0.25 <= new_scale <= 4.0:
            self.view_scale = new_scale
        else:
            return

        # keep previous offsets ...
        scroll_offset_y = self.container_images.v_scroll.value if self.container_images.v_scroll.active else 0
        scroll_offset_x = self.container_images.h_scroll.value if self.container_images.h_scroll.active else 0

        prev_center_y = scroll_offset_y + self.container_images.height / 2
        prev_center_x = scroll_offset_x + self.container_images.width / 2

        # compute new scroll bar offsets
        scale_factor = (new_scale / prev_scale)
        new_off_y = prev_center_y * scale_factor - self.container_images.height / 2
        new_off_x = prev_center_x * scale_factor - self.container_images.width / 2

        # update view ....
        self.update_current_view(True)

        # set offsets
        if self.container_images.v_scroll.active and 0 <= new_off_y <= self.container_images.v_scroll.max:
            self.container_images.v_scroll.value = new_off_y
        if self.container_images.h_scroll.active and 0 <= new_off_x <= self.container_images.h_scroll.max:
            self.container_images.h_scroll.value = new_off_x

        # if selection rectangle is active ...
        if self.edition_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection:
            self.canvas_select.elements["selection_rectangle"].x *= scale_factor
            self.canvas_select.elements["selection_rectangle"].y *= scale_factor
            self.canvas_select.elements["selection_rectangle"].w *= scale_factor
            self.canvas_select.elements["selection_rectangle"].h *= scale_factor

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_zoom_reduce_click(self, button):
        self.update_view_scale(self.view_scale - 0.25)

    def btn_zoom_increase_click(self, button):
        self.update_view_scale(self.view_scale + 0.25)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def greedy_matching_scores(self):
        sel_rect = self.canvas_select.elements["selection_rectangle"]
        rect_x = int(round(sel_rect.x / self.view_scale))
        rect_y = int(round(sel_rect.y / self.view_scale))
        rect_w = int(round(sel_rect.w / self.view_scale))
        rect_h = int(round(sel_rect.h / self.view_scale))

        # identify CC's from current frame within selected region (containment)
        curr_kf = self.keyframe_annotations[self.selected_keyframe]
        curr_ccs = curr_kf.ccs_in_region(rect_x, rect_x + rect_w, rect_y, rect_y + rect_h)
        curr_ccs = {cc.strID(): cc for cc in curr_ccs}
        # only keep those that have not been matched yet ...
        filtered_ccs = {}
        for curr_cc_str_id in curr_ccs:
            if self.cc_group[self.selected_keyframe][curr_cc_str_id].start_frame == self.selected_keyframe:
                # unmatched CC ...
                filtered_ccs[curr_cc_str_id] = curr_ccs[curr_cc_str_id]

        print("Total candidates (C-KF): " + str(len(curr_ccs)))
        if len(filtered_ccs) != len(curr_ccs):
            print("Total candidates not previously matched (C-KF): " + str(len(filtered_ccs)))

            curr_ccs = filtered_ccs

        # identify CC's from prev frame within selected region (containment)
        prev_kf = self.keyframe_annotations[self.selected_keyframe - 1]

        prev_ccs = prev_kf.ccs_in_region(rect_x - self.matching_delta_x, rect_x - self.matching_delta_x + rect_w,
                                        rect_y - self.matching_delta_y, rect_y - self.matching_delta_y + rect_h)
        prev_ccs = {cc.strID(): ConnectedComponent.ShallowCopy(cc) for cc in prev_ccs}
        # modify box using delta
        for prev_cc_str_id in prev_ccs:
            prev_ccs[prev_cc_str_id].translateBox(self.matching_delta_x, self.matching_delta_y)

        # compute all scores
        all_matches = []
        for curr_cc_str_id in curr_ccs:
            curr_cc = curr_ccs[curr_cc_str_id]
            for prev_cc_str_id in prev_ccs:
                prev_cc = prev_ccs[prev_cc_str_id]

                if (curr_cc.min_x < prev_cc.max_x and prev_cc.min_x < curr_cc.max_x and
                    curr_cc.min_y < prev_cc.max_y and prev_cc.min_y < curr_cc.max_y):
                    recall, precision = curr_cc.getOverlapFMeasure(prev_cc, False, False)

                    all_matches.append((recall, precision, prev_cc, curr_cc))

        # restore original box
        for prev_cc_str_id in prev_ccs:
            prev_ccs[prev_cc_str_id].translateBox(-self.matching_delta_x, -self.matching_delta_y)

        # sort by decreasing recall
        all_matches = sorted(all_matches, reverse=True, key=lambda x:x[0])

        # now, greedily pick 1 to 1 matches ...
        self.matching_scores = []
        matched_curr_ids = {}
        matched_prev_ids = {}
        for recall, precision, prev_cc, curr_cc in all_matches:
            prev_cc_str_id = prev_cc.strID()
            curr_cc_str_id = curr_cc.strID()

            # filter if already matched ....
            if prev_cc_str_id in matched_prev_ids:
                continue
            if curr_cc_str_id in matched_curr_ids:
                continue

            # accept match ....
            #print((prev_cc, curr_cc))
            self.matching_scores.append((recall, precision, prev_cc, curr_cc))
            # marked as matched ...
            matched_prev_ids[prev_cc_str_id] = True
            matched_curr_ids[curr_cc_str_id] = True


    def btn_confirm_accept_click(self, button):
        if self.edition_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection:
            # compute potential matches ...
            self.greedy_matching_scores()
            # clear base image where matches will be shown ...
            gray_mask = self.base_matching.sum(axis=2) < 255 * 3
            self.base_matching[gray_mask, 0] = 128
            self.base_matching[gray_mask, 1] = 128
            self.base_matching[gray_mask, 2] = 128
            # move to the next stage ..
            self.set_editor_mode(GTUniqueCCAnnotator.ModeMatch_Matching)
            self.update_current_view(False)

        elif self.edition_mode == GTUniqueCCAnnotator.ModeMatch_Matching:
            # accept matches
            for precision, recall, prev_cc, curr_cc in self.matching_scores:
                if precision >= self.matching_min_precision and recall >= self.matching_min_recall:
                    # merge to previous group ...
                    prev_id = prev_cc.strID()
                    prev_group = self.cc_group[self.selected_keyframe - 1][prev_id]

                    curr_id = curr_cc.strID()
                    curr_group = self.cc_group[self.selected_keyframe][curr_id]

                    # for each member of the current group ...
                    for kf_offset, cc in enumerate(curr_group.cc_refs):
                        # make element point to previous group
                        cc_id = cc.strID()
                        #print(self.cc_group[self.selected_keyframe + kf_offset][cc_id].start_frame)
                        self.cc_group[self.selected_keyframe + kf_offset][cc_id] = prev_group
                        #print(self.cc_group[self.selected_keyframe + kf_offset][cc_id].start_frame)

                        # add element in current group to previous group
                        prev_group.cc_refs.append(cc)

                    # remove group from list of unique groups
                    self.unique_groups.remove(curr_group)

            # update count ....
            self.lbl_cc_unique.set_text("Total Unique CC:\n" + str(len(self.unique_groups)))

            print("PENDING UNDO/REDO")

            # update colored images ...
            self.update_colored_cache(self.selected_keyframe)

            self.set_editor_mode(GTUniqueCCAnnotator.ModeNavigate)
            self.update_current_view(False)

        elif self.edition_mode == GTUniqueCCAnnotator.ModeMatch_Remove:
            sel_rect = self.canvas_select.elements["selection_rectangle"]
            rect_x = int(round(sel_rect.x / self.view_scale))
            rect_y = int(round(sel_rect.y / self.view_scale))
            rect_w = int(round(sel_rect.w / self.view_scale))
            rect_h = int(round(sel_rect.h / self.view_scale))

            # identify CC's from current frame within selected region (containment)
            curr_kf = self.keyframe_annotations[self.selected_keyframe]
            curr_ccs = curr_kf.ccs_in_region(rect_x, rect_x + rect_w, rect_y, rect_y + rect_h)
            curr_ccs = {cc.strID(): cc for cc in curr_ccs}
            # only keep those that have been previously matched
            filtered_ccs = {}
            for curr_cc_str_id in curr_ccs:
                if self.cc_group[self.selected_keyframe][curr_cc_str_id].start_frame < self.selected_keyframe:
                    # matched CC ...
                    filtered_ccs[curr_cc_str_id] = curr_ccs[curr_cc_str_id]

            print("Total CC in region (C-KF): " + str(len(curr_ccs)))
            print("Total matches to remove (C-KF): " + str(len(filtered_ccs)))
            curr_ccs = filtered_ccs

            # Remove CC's from group (split) and add their own group
            for curr_cc_str_id in curr_ccs:
                # previous group
                prev_group = self.cc_group[self.selected_keyframe][curr_cc_str_id]

                # ask the group to split
                new_group = UniqueCCGroup.Split(prev_group, self.selected_keyframe)
                # link CCs on the new group to the new group
                for split_offset, split_cc in enumerate(new_group.cc_refs):
                    split_cc_str_id = split_cc.strID()
                    self.cc_group[self.selected_keyframe + split_offset][split_cc_str_id] = new_group

                # add new group to the list of unique CC
                self.unique_groups.append(new_group)

            print("Pending to be able to UNDO/REDO")

            # update colored images ...
            self.update_colored_cache(self.selected_keyframe)

            self.set_editor_mode(GTUniqueCCAnnotator.ModeNavigate)
            self.update_current_view(False)

    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        self.set_editor_mode(GTUniqueCCAnnotator.ModeNavigate)
        self.update_current_view(False)

    def btn_view_raw_click(self, button):
        self.view_mode = GTUniqueCCAnnotator.ViewModeRaw
        self.update_current_view()

    def btn_view_gray_click(self, button):
        self.view_mode = GTUniqueCCAnnotator.ViewModeGray
        self.update_current_view()

    def btn_view_binary_click(self, button):
        self.view_mode = GTUniqueCCAnnotator.ViewModeBinary
        self.update_current_view()

    def btn_view_colored_click(self, button):
        self.view_mode = GTUniqueCCAnnotator.ViewModeColored
        self.update_current_view()

    def prepare_selection_rectangle(self, margin):
        # Default selection rectangle is relative to current view
        if self.container_images.v_scroll.active:
            rect_y = margin + self.container_images.v_scroll.value
            rect_h = self.container_images.height - (margin * 2) - self.container_images.h_scroll.height
        else:
            rect_y = margin
            rect_h = self.img_main.height - margin * 2

        if self.container_images.h_scroll.active:
            rect_x = margin + self.container_images.h_scroll.value
            rect_w = self.container_images.width - margin * 2 - self.container_images.v_scroll.width
        else:
            rect_x = margin
            rect_w = self.img_main.width - margin * 2

        self.canvas_select.elements["selection_rectangle"].x = rect_x
        self.canvas_select.elements["selection_rectangle"].y = rect_y
        self.canvas_select.elements["selection_rectangle"].w = rect_w
        self.canvas_select.elements["selection_rectangle"].h = rect_h

    def btn_matches_add_click(self, button):
        if self.selected_keyframe == 0:
            print("Cannot match elements on the first frame")
            return

        self.prepare_selection_rectangle(40)
        self.set_editor_mode(GTUniqueCCAnnotator.ModeMatch_RegionSelection)
        self.update_matching_image()
        self.update_current_view(False)

    def btn_matches_del_click(self, button):
        if self.selected_keyframe == 0:
            print("There are no matches on the first frame")
            return

        self.prepare_selection_rectangle(40)
        self.set_editor_mode(GTUniqueCCAnnotator.ModeMatch_Remove)
        self.update_current_view(False)

    def btn_undo_click(self, button):
        print("Not yet available!")
        pass

    def btn_redo_click(self, button):
        print("Not yet available!")
        pass

    def btn_save_click(self, button):
        xml_str = UniqueCCGroup.GenerateGroupsXML(self.keyframe_annotations, self.unique_groups)

        unique_cc_filename = self.output_path + "/unique_ccs.xml"
        out_file = open(unique_cc_filename, "w")
        out_file.write(xml_str)
        out_file.close()

        print("Saved to: " + unique_cc_filename)


    def btn_exit_click(self, button):
        if len(self.undo_stack) > 0:
            # confirm before losing changes
            self.set_editor_mode(GTUniqueCCAnnotator.ModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")

    def update_matching_image(self):
        prev_binary = self.keyframe_annotations[self.selected_keyframe - 1].binary_image[:, :, 0]
        curr_binary = self.keyframe_annotations[self.selected_keyframe].binary_image[:, :, 0]

        self.base_matching = Visualizer.combine_bin_images_w_disp(curr_binary, prev_binary, self.matching_delta_x,
                                                                  self.matching_delta_y, 0)

    def scroll_delta_x_change(self, scroll):
        self.matching_delta_x = int(scroll.value)
        self.lbl_delta_x.set_text("Delta X: " + str(self.matching_delta_x))
        self.update_matching_image()
        self.update_current_view()

    def scroll_delta_y_change(self, scroll):
        self.matching_delta_y = int(scroll.value)
        self.lbl_delta_y.set_text("Delta Y: " + str(self.matching_delta_y))
        self.update_matching_image()
        self.update_current_view()


    def scroll_min_recall_change(self, scroll):
        self.matching_min_recall = scroll.value / 100.0
        self.lbl_min_recall.set_text("Minimum Recall: " + str(int(self.matching_min_recall * 100)))
        self.update_current_view()

    def scroll_min_precision_change(self, scroll):
        self.matching_min_precision = scroll.value / 100.0
        self.lbl_min_precision.set_text("Minimum Precision: " + str(int(self.matching_min_precision * 100)))
        self.update_current_view()


    def set_editor_mode(self, new_mode):
        self.edition_mode = new_mode
        self.container_nav_buttons.visible = (new_mode == GTUniqueCCAnnotator.ModeNavigate)

        self.container_confirm_buttons.visible = (new_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection or
                                                  new_mode == GTUniqueCCAnnotator.ModeMatch_Matching or
                                                  new_mode == GTUniqueCCAnnotator.ModeMatch_Remove or
                                                  new_mode == GTUniqueCCAnnotator.ModeExitConfirm)

        if new_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection:
            self.lbl_confirm_message.set_text("Selecting Matching Region")
        elif new_mode == GTUniqueCCAnnotator.ModeMatch_Matching:
            self.lbl_confirm_message.set_text("Matching Unique CCs")
        elif new_mode == GTUniqueCCAnnotator.ModeMatch_Remove:
            self.lbl_confirm_message.set_text("Remove CCs Matches")
        elif new_mode == GTUniqueCCAnnotator.ModeExitConfirm:
            self.lbl_confirm_message.set_text("Exit Without Saving?")

        if new_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection or new_mode == GTUniqueCCAnnotator.ModeMatch_Remove:
            # show rectangle
            self.canvas_select.locked = False
            self.canvas_select.elements["selection_rectangle"].visible = True
        else:
            # for every other mode
            self.canvas_select.locked = True
            self.canvas_select.elements["selection_rectangle"].visible = False

        self.container_state_buttons.visible = (new_mode == GTUniqueCCAnnotator.ModeNavigate)
        self.container_stats.visible = (new_mode == GTUniqueCCAnnotator.ModeNavigate)
        self.container_action_buttons.visible = (new_mode == GTUniqueCCAnnotator.ModeNavigate)

        self.container_matching_translation.visible = (new_mode == GTUniqueCCAnnotator.ModeMatch_RegionSelection)
        self.container_matching_strictness.visible = (new_mode == GTUniqueCCAnnotator.ModeMatch_Matching)
