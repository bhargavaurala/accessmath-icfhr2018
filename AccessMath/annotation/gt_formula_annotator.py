
import os

import cv2
import numpy as np
import pygame

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.annotation.formula_ccs import FormulaCCs
from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.unique_cc_group import UniqueCCGroup
from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_canvas import ScreenCanvas
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_image import ScreenImage
from AccessMath.interface.controls.screen_label import ScreenLabel


class GTFormulaAnnotator(Screen):
    ModeNavigate = 0
    ModeFormula_Add = 1
    ModeFormula_Edit = 2
    ModeFormula_TagEdit = 3
    ModeFormula_Remove = 4
    ModeExit_Confirm = 5

    ViewModeRaw = 0
    ViewModeGray = 1
    ViewModeBinary = 2
    ViewModeColored = 3

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Formula Ground Truth Annotation Interface", size)

        general_background = (50, 80, 160)
        text_color = (255, 255, 255)
        button_text_color = (20, 20, 50)
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
            raise Exception("No unique CC data found for lecture. Must label Unique CC first")

        self.view_mode = GTFormulaAnnotator.ViewModeColored
        self.edition_mode = GTFormulaAnnotator.ModeNavigate
        self.view_scale = 1.0
        self.selected_keyframe = 0

        self.selected_formula = 0
        self.adding_groups = []
        self.formulas_per_frame = [[] for idx in range(len(self.keyframe_annotations))]

        saved_filename = self.output_path + "/formula_ccs.xml"
        if os.path.exists(saved_filename):
            # load saved data ...
            self.formulas_ccs = FormulaCCs.FormulasFromXML(self.unique_groups, saved_filename)
            # add to index per frame ...
            for formula in self.formulas_ccs:
                for frame_idx in range(formula.first_visible, formula.last_visible + 1):
                    self.formulas_per_frame[frame_idx].append(formula)

            print("Loaded: " + saved_filename)
        else:
            self.formulas_ccs = []

        # update draw cache of highlighted formulae ...
        self.colored_cache = [None] * len(self.keyframe_annotations)
        for idx in range(len(self.keyframe_annotations)):
            self.update_colored_cache(idx)

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

        # ================================================================================

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

        # ================================================================================

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

        # =================================================================================

        # Panel with action buttons (Add/Remove links)
        self.container_action_buttons = ScreenContainer("container_action_buttons", (container_width, 240),
                                                        general_background)
        self.container_action_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_action_buttons)

        self.lbl_nav_formula = ScreenLabel("lbl_nav_formula", "X / X", 21,
                                           container_width - 10, 1)
        self.lbl_nav_formula.position = (5, 5)
        self.lbl_nav_formula.set_background(general_background)
        self.lbl_nav_formula.set_color(text_color)
        self.container_action_buttons.append(self.lbl_nav_formula)

        self.btn_nav_formula_prev = ScreenButton("btn_nav_formula_prev", "Prev", 21, button_2_width)
        self.btn_nav_formula_prev.set_colors(button_text_color, button_back_color)
        self.btn_nav_formula_prev.position = (button_2_left, self.lbl_nav_formula.get_bottom() + 10)
        self.btn_nav_formula_prev.click_callback = self.btn_nav_formula_prev_click
        self.container_action_buttons.append(self.btn_nav_formula_prev)

        self.btn_nav_formula_next = ScreenButton("btn_nav_formula_next", "Next", 21, button_2_width)
        self.btn_nav_formula_next.set_colors(button_text_color, button_back_color)
        self.btn_nav_formula_next.position = (button_2_right, self.lbl_nav_formula.get_bottom() + 10)
        self.btn_nav_formula_next.click_callback = self.btn_nav_formula_next_click
        self.container_action_buttons.append(self.btn_nav_formula_next)

        self.btn_formulas_add = ScreenButton("btn_formulas_add", "Add Formula", 21, button_2_width)
        self.btn_formulas_add.set_colors(button_text_color, button_back_color)
        self.btn_formulas_add.position = (button_2_left, self.btn_nav_formula_next.get_bottom() + 20)
        self.btn_formulas_add.click_callback = self.btn_formulas_add_click
        self.container_action_buttons.append(self.btn_formulas_add)

        self.btn_formulas_del = ScreenButton("btn_formulas_del", "Del. Formula", 21, button_2_width)
        self.btn_formulas_del.set_colors(button_text_color, button_back_color)
        self.btn_formulas_del.position = (button_2_right, self.btn_nav_formula_next.get_bottom() + 20)
        self.btn_formulas_del.click_callback = self.btn_formulas_del_click
        self.container_action_buttons.append(self.btn_formulas_del)

        self.btn_formula_update_tag = ScreenButton("btn_formula_update_tag", "Update Tag", 21, button_width)
        self.btn_formula_update_tag.set_colors(button_text_color, button_back_color)
        self.btn_formula_update_tag.position = (button_left, self.btn_formulas_del.get_bottom() + 20)
        self.btn_formula_update_tag.click_callback = self.btn_formula_update_tag_click
        self.container_action_buttons.append(self.btn_formula_update_tag)

        self.lbl_formula_tag = ScreenLabel("lbl_formula_tag", "Tag: ?", 21, container_width - 10, 1)
        self.lbl_formula_tag.position = (5, self.btn_formula_update_tag.get_bottom() + 20)
        self.lbl_formula_tag.set_background(general_background)
        self.lbl_formula_tag.set_color(text_color)
        self.container_action_buttons.append(self.lbl_formula_tag)

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
        self.img_main.mouse_button_down_callback = self.img_mouse_down
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

        self.update_selected_formula(0)

        self.update_current_view(True)

    def update_colored_cache(self, update_frame):
        keyframe = self.keyframe_annotations[update_frame]

        # start with a copy of the binary frame in 3 channels
        colored_image = keyframe.binary_image.copy()

        # now, color the CCs that belong to any formula in blue ...
        for formula in self.formulas_per_frame[update_frame]:
            assert isinstance(formula, FormulaCCs)

            self.highlight_groups(colored_image, formula.groups_refs, update_frame, (0, 0, 255))

        self.colored_cache[update_frame] = colored_image

    def highlight_groups(self, colored_image, group_refs, base_frame, color):
        for group in group_refs:
            # find image on the corresponding frame ...
            cc_ref = group.cc_refs[base_frame - group.start_frame]

            # mark formula CC in blue ...
            cc_cut = colored_image[cc_ref.min_y:cc_ref.max_y + 1, cc_ref.min_x:cc_ref.max_x + 1, :]
            cc_mask = cc_ref.img > 0
            cc_cut[cc_mask, 0] = color[0]
            cc_cut[cc_mask, 1] = color[1]
            cc_cut[cc_mask, 2] = color[2]

    def update_current_view(self, resized=False):
        if self.view_mode == GTFormulaAnnotator.ViewModeGray:
            base_image = self.keyframe_annotations[self.selected_keyframe].grayscale_image
        elif self.view_mode == GTFormulaAnnotator.ViewModeBinary:
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        elif self.view_mode == GTFormulaAnnotator.ViewModeColored:
            base_image = self.colored_cache[self.selected_keyframe]
        else:
            # raw ...
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image

        h, w, c = base_image.shape
        modified_image = base_image.copy()

        if self.edition_mode == GTFormulaAnnotator.ModeNavigate:
            # highlight selected formula ... if any and visible ...
            if self.selected_formula < len(self.formulas_ccs):
                formula = self.formulas_ccs[self.selected_formula]

                if formula.visible_at(self.selected_keyframe):
                    self.highlight_groups(modified_image, formula.groups_refs, self.selected_keyframe, (255, 0, 0))

        elif self.edition_mode == GTFormulaAnnotator.ModeFormula_Add:
            self.highlight_groups(modified_image, self.adding_groups, self.selected_keyframe, (0, 255, 0))

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

    def update_selected_formula(self, new_formula):
        if len(self.formulas_ccs) == 0:
            # no formula on the list ...
            self.lbl_nav_formula.set_text("Formula: 0 / 0")
            self.lbl_formula_tag.set_text("Tag: ")
        else:
            if new_formula < 0:
                new_formula = 0
            if new_formula >= len(self.formulas_ccs):
                new_formula = len(self.formulas_ccs) - 1

            self.selected_formula = new_formula

            current_formula = self.formulas_ccs[self.selected_formula]

            self.lbl_nav_formula.set_text("Formula: {0:d} / {1:d}".format(self.selected_formula + 1, len(self.formulas_ccs)))
            self.lbl_formula_tag.set_text("Tag: " + current_formula.latex_tag)

            self.update_selected_keyframe(current_formula.first_visible)

    def btn_nav_formula_prev_click(self, button):
        self.update_selected_formula(self.selected_formula - 1)

    def btn_nav_formula_next_click(self, button):
        self.update_selected_formula(self.selected_formula + 1)

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

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_zoom_reduce_click(self, button):
        self.update_view_scale(self.view_scale - 0.25)

    def btn_zoom_increase_click(self, button):
        self.update_view_scale(self.view_scale + 0.25)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def add_new_formula(self, unique_cc_groups, latex_tag):
        # create formula ...
        new_formula = FormulaCCs(unique_cc_groups, latex_tag)

        # add to the frames ...
        for frame_idx in range(new_formula.first_visible, new_formula.last_visible + 1):
            self.formulas_per_frame[frame_idx].append(new_formula)
            # update the base image of that frame ...
            self.update_colored_cache(frame_idx)

        # add to the list ...
        self.formulas_ccs.append(new_formula)

    def btn_confirm_accept_click(self, button):
        if self.edition_mode == GTFormulaAnnotator.ModeExit_Confirm:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")
        elif self.edition_mode == GTFormulaAnnotator.ModeFormula_Add:
            # if formula elements have been selected ... then add ...
            self.add_new_formula(self.adding_groups, "")
            self.adding_groups = []

            # update GUI ...
            self.update_selected_formula(len(self.formulas_ccs) - 1)
            self.set_editor_mode(GTFormulaAnnotator.ModeNavigate)
            self.update_current_view(False)


    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        self.set_editor_mode(GTFormulaAnnotator.ModeNavigate)
        self.update_current_view(False)

    def btn_view_raw_click(self, button):
        self.view_mode = GTFormulaAnnotator.ViewModeRaw
        self.update_current_view()

    def btn_view_gray_click(self, button):
        self.view_mode = GTFormulaAnnotator.ViewModeGray
        self.update_current_view()

    def btn_view_binary_click(self, button):
        self.view_mode = GTFormulaAnnotator.ViewModeBinary
        self.update_current_view()

    def btn_view_colored_click(self, button):
        self.view_mode = GTFormulaAnnotator.ViewModeColored
        self.update_current_view()

    def btn_formulas_add_click(self, button):
        self.set_editor_mode(GTFormulaAnnotator.ModeFormula_Add)
        self.update_current_view(False)

    def btn_formulas_del_click(self, button):
        if len(self.formulas_ccs) > 0:
            to_delete = self.formulas_ccs[self.selected_formula]

            for frame_idx in range(to_delete.first_visible, to_delete.last_visible + 1):
                self.formulas_per_frame[frame_idx].remove(to_delete)
                self.update_colored_cache(frame_idx)

            self.formulas_ccs.remove(to_delete)
            self.update_selected_formula(self.selected_formula)
            self.update_current_view(False)

    def btn_formula_update_tag_click(self, button):
        if len(self.formulas_ccs) > 0:
            pygame.display.iconify()
            new_tag = input("Enter new tag: ")
            self.formulas_ccs[self.selected_formula].latex_tag = new_tag
            self.update_selected_formula(self.selected_formula)

    def btn_undo_click(self, button):
        print("Not yet available!")
        pass

    def btn_redo_click(self, button):
        print("Not yet available!")
        pass

    def btn_save_click(self, button):
        xml_str = FormulaCCs.GenerateFormulaXML(self.formulas_ccs)

        formula_ccs_filename = self.output_path + "/formula_ccs.xml"
        out_file = open(formula_ccs_filename, "w")
        out_file.write(xml_str)
        out_file.close()

        print("Saved to: " + formula_ccs_filename)

        self.undo_stack = []
        self.redo_stack = []


    def btn_exit_click(self, button):
        if len(self.undo_stack) > 0:
            # confirm before losing changes
            self.set_editor_mode(GTFormulaAnnotator.ModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")

    def set_editor_mode(self, new_mode):
        self.edition_mode = new_mode

        self.container_nav_buttons.visible = (new_mode == GTFormulaAnnotator.ModeNavigate)
        self.container_confirm_buttons.visible = (new_mode == GTFormulaAnnotator.ModeFormula_Add)

        if new_mode == GTFormulaAnnotator.ModeFormula_Add:
            self.lbl_confirm_message.set_text("Add Formula")
        elif new_mode == GTFormulaAnnotator.ModeExit_Confirm:
            self.lbl_confirm_message.set_text("Exit Without Saving?")
        elif new_mode == GTFormulaAnnotator.ModeFormula_Add:
            pass

        self.canvas_select.locked = True
        self.canvas_select.elements["selection_rectangle"].visible = False

        self.container_state_buttons.visible = (new_mode == GTFormulaAnnotator.ModeNavigate)
        self.container_stats.visible = (new_mode == GTFormulaAnnotator.ModeNavigate)
        self.container_action_buttons.visible = (new_mode == GTFormulaAnnotator.ModeNavigate)

    def img_mouse_down(self, img_object, pos, button):
        if button == 1:
            # ... first, get click location on original image space
            scaled_x, scaled_y = pos
            click_x = int(scaled_x / self.view_scale)
            click_y = int(scaled_y / self.view_scale)

            if self.edition_mode == GTFormulaAnnotator.ModeFormula_Add:
                # determine which CC was clicked ...
                cc_found = None
                for cc in self.keyframe_annotations[self.selected_keyframe].binary_cc:
                    if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                        if cc.img[click_y - cc.min_y, click_x - cc.min_x] == 255:
                            cc_found = cc
                            break

                if cc_found is not None:
                    current_group = self.cc_group[self.selected_keyframe][cc_found.strID()]

                    if current_group in self.adding_groups:
                        self.adding_groups.remove(current_group)
                    else:
                        self.adding_groups.append(current_group)

                    self.update_current_view(False)

