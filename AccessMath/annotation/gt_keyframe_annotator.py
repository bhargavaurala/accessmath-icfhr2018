
import math
import os

import cv2
import numpy as np

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.annotation.gt_binary_annotator import GTBinaryAnnotator
from AccessMath.annotation.gt_pixel_binary_annotator import GTPixelBinaryAnnotator
from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.keyframe_portion import KeyFramePortion
from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_canvas import ScreenCanvas
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AccessMath.interface.controls.screen_image import ScreenImage
from AccessMath.interface.controls.screen_label import ScreenLabel


class GTKeyFrameAnnotator(Screen):
    EditionModeNavigate = 0
    EditionModeAddPortion = 1
    EditionModeClearRegion = 2
    EditionModeExitConfirm = 3

    ViewModeRaw = 0
    ViewModeGray = 1
    ViewModeBinary = 2
    ViewModeCombined = 3

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Key-Frame Ground Truth Annotation Interface", size)

        general_background = (40, 125, 20)
        text_color = (255, 255, 255)
        button_text_color = (35, 50, 20)
        button_back_color = (228, 228, 228)
        self.elements.back_color = general_background

        self.db_name = db_name
        self.lecture_title = lecture_title

        self.output_path = output_path

        export_filename = self.output_path + "/segments.xml"
        export_image_prefix = self.output_path + "/keyframes/"
        self.keyframe_annotations = KeyFrameAnnotation.LoadExportedKeyframes(export_filename, export_image_prefix)

        portions_filename = self.output_path + "/portions.xml"
        portions_path = self.output_path + "/portions/"
        if os.path.exists(portions_filename):
            # Saved data detected, loading
            print("Previously saved portion data detected, loading")
            KeyFrameAnnotation.LoadKeyframesPortions(portions_filename, self.keyframe_annotations, portions_path)
        else:
            print("No previously saved portion data detected")

        #self.output_prefix = output_prefix

        self.view_mode = GTKeyFrameAnnotator.ViewModeRaw
        self.edition_mode = GTKeyFrameAnnotator.EditionModeNavigate
        self.show_portions = False
        self.view_scale = 1.0
        self.selected_keyframe = 0
        self.selected_portion = [0 for idx in range(len(self.keyframe_annotations))]
        self.min_highlight_size = 0
        self.editing_portion = None

        if len(self.keyframe_annotations) > 0:
            print("Key-frames Loaded: " + str(len(self.keyframe_annotations)))
        else:
            raise Exception("Cannot start with 0 key-frames")

        # add elements....
        # TITLE
        label_title = ScreenLabel("title", "ACCESS MATH - Video Keyframe Annotation Tool", 28)
        label_title.background = general_background
        label_title.position = (int((self.width - label_title.width) / 2), 20)
        label_title.set_color(text_color)

        self.small_mode = self.height < 800

        if not self.small_mode:
            self.elements.append(label_title)

        container_top = 10 + (label_title.get_bottom() if not self.small_mode else 0)

        # Navigation panel to move accross frames
        self.container_nav_buttons = ScreenContainer("container_nav_buttons", (300, 70), back_color=general_background)
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
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (300, 70), back_color=general_background)
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
        self.container_view_buttons = ScreenContainer("container_view_buttons", (300, 320), back_color=general_background)
        self.container_view_buttons.position = (self.width - self.container_view_buttons.width - 10,
                                                self.container_nav_buttons.get_bottom() + 10)
        self.elements.append(self.container_view_buttons)


        button_width = 190
        button_left = (self.container_view_buttons.width - button_width) / 2

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", 21, 290, 1)
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

        self.btn_view_raw = ScreenButton("btn_view_raw", "Raw View", 21, button_width)
        self.btn_view_raw.set_colors(button_text_color, button_back_color)
        self.btn_view_raw.position = (button_left, self.btn_zoom_zero.get_bottom() + 20)
        self.btn_view_raw.click_callback = self.btn_view_raw_click
        self.container_view_buttons.append(self.btn_view_raw)

        self.btn_view_gray = ScreenButton("btn_view_gray", "Grayscale View", 21, button_width)
        self.btn_view_gray.set_colors(button_text_color, button_back_color)
        self.btn_view_gray.position = (button_left, self.btn_view_raw.get_bottom() + 10)
        self.btn_view_gray.click_callback = self.btn_view_gray_click
        self.container_view_buttons.append(self.btn_view_gray)

        self.btn_view_binary = ScreenButton("btn_view_binary", "Binary View", 21, button_width)
        self.btn_view_binary.set_colors(button_text_color, button_back_color)
        self.btn_view_binary.position = (button_left, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_binary.click_callback = self.btn_view_binary_click
        self.container_view_buttons.append(self.btn_view_binary)

        self.btn_view_combined = ScreenButton("btn_view_combined", "Combined View", 21, button_width)
        self.btn_view_combined.set_colors(button_text_color, button_back_color)
        self.btn_view_combined.position = (button_left, self.btn_view_binary.get_bottom() + 10)
        self.btn_view_combined.click_callback = self.btn_view_combined_click
        self.container_view_buttons.append(self.btn_view_combined)

        self.lbl_small_highlight = ScreenLabel("lbl_small_highlight", "Highlight CCs smaller than: " + str(self.min_highlight_size), 21, 290, 1)
        self.lbl_small_highlight.position = (5, self.btn_view_combined.get_bottom() + 20)
        self.lbl_small_highlight.set_background(general_background)
        self.lbl_small_highlight.set_color(text_color)
        self.container_view_buttons.append(self.lbl_small_highlight)

        self.highlight_scroll = ScreenHorizontalScroll("threshold_scroll", 0, 100, 0, 10)
        self.highlight_scroll.position = (5, self.lbl_small_highlight.get_bottom() + 10)
        self.highlight_scroll.width = 290
        self.highlight_scroll.scroll_callback = self.highlight_scroll_change
        self.container_view_buttons.append(self.highlight_scroll)

        # Panel with state buttons (Undo, Redo, Save)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (300, 200), general_background)
        self.container_state_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 10)
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

        image_width = self.width - self.container_view_buttons.width - 30
        image_height = self.height - container_top - 70
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, container_top)
        self.elements.append(self.container_images)

        # ... image objects ...
        tempo_blank = np.zeros((50, 50, 3), np.uint8)
        tempo_blank[:, :, :] = 255
        self.img_main = ScreenImage("img_raw", tempo_blank, 0, 0, True, cv2.INTER_NEAREST)
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

        self.container_portions = ScreenContainer("container_portions", (self.container_images.width, 50), back_color=general_background)
        self.container_portions.position = (10, self.container_images.get_bottom() + 10)
        self.elements.append(self.container_portions)

        self.lbl_portions = ScreenLabel("lbl_portions", "Portions: " + str(len(self.keyframe_annotations[self.selected_keyframe].portions)), 21, 120, 0)
        self.lbl_portions.position = (5, 15)
        self.lbl_portions.set_background(general_background)
        self.lbl_portions.set_color(text_color)
        self.container_portions.append(self.lbl_portions)

        button_width = 65 if self.small_mode else 90
        font_size = 18 if self.small_mode else 21

        self.btn_portions_add = ScreenButton("btn_portions_add", "Add", font_size, button_width)
        self.btn_portions_add.set_colors(button_text_color, button_back_color)
        self.btn_portions_add.position = (self.lbl_portions.get_right() + 20, 5)
        self.btn_portions_add.click_callback = self.btn_portions_add_click
        self.container_portions.append(self.btn_portions_add)

        self.btn_portions_show = ScreenButton("btn_portions_show", "Show", font_size, button_width)
        self.btn_portions_show.set_colors(button_text_color, button_back_color)
        self.btn_portions_show.position = (self.btn_portions_add.get_right() + 10, 5)
        self.btn_portions_show.click_callback = self.btn_portions_show_click
        self.container_portions.append(self.btn_portions_show)

        self.btn_portions_hide = ScreenButton("btn_portions_hide", "Hide", font_size, button_width)
        self.btn_portions_hide.set_colors(button_text_color, button_back_color)
        self.btn_portions_hide.position = (self.btn_portions_add.get_right() + 10, 5)
        self.btn_portions_hide.click_callback = self.btn_portions_hide_click
        self.container_portions.append(self.btn_portions_hide)

        self.btn_portions_edit = ScreenButton("btn_portions_edit", "Edit", font_size, button_width)
        self.btn_portions_edit.set_colors(button_text_color, button_back_color)
        self.btn_portions_edit.position = (self.btn_portions_show.get_right() + 10, 5)
        self.btn_portions_edit.click_callback = self.btn_portions_edit_click
        self.container_portions.append(self.btn_portions_edit)

        self.btn_portions_prev = ScreenButton("btn_portions_prev", "Prev", font_size, button_width)
        self.btn_portions_prev.set_colors(button_text_color, button_back_color)
        self.btn_portions_prev.position = (self.btn_portions_edit.get_right() + 10, 5)
        self.btn_portions_prev.click_callback = self.btn_portions_prev_click
        self.container_portions.append(self.btn_portions_prev)

        self.btn_portions_next = ScreenButton("btn_portions_next", "Next", font_size, button_width)
        self.btn_portions_next.set_colors(button_text_color, button_back_color)
        self.btn_portions_next.position = (self.btn_portions_prev.get_right() + 10, 5)
        self.btn_portions_next.click_callback = self.btn_portions_next_click
        self.container_portions.append(self.btn_portions_next)

        self.btn_portions_del = ScreenButton("btn_portions_del", "Del", font_size, button_width)
        self.btn_portions_del.set_colors(button_text_color, button_back_color)
        self.btn_portions_del.position = (self.btn_portions_next.get_right() + 20, 5)
        self.btn_portions_del.click_callback = self.btn_portions_del_click
        self.container_portions.append(self.btn_portions_del)

        clear_text = "C. Region" if self.small_mode else "Clear Region"
        self.btn_portions_clear = ScreenButton("btn_portions_clear", clear_text, font_size, button_width + 30)
        self.btn_portions_clear.set_colors(button_text_color, button_back_color)
        self.btn_portions_clear.position = (self.btn_portions_del.get_right() + 10, 5)
        self.btn_portions_clear.click_callback = self.btn_portions_clear_click
        self.container_portions.append(self.btn_portions_clear)

        copy_text = "CP. Next" if self.small_mode else "Copy to Next"
        self.btn_portions_copy_next = ScreenButton("btn_portions_copy_next", copy_text, font_size, button_width + 30)
        self.btn_portions_copy_next.set_colors(button_text_color, button_back_color)
        self.btn_portions_copy_next.position = (self.btn_portions_clear.get_right() + 10, 5)
        self.btn_portions_copy_next.click_callback = self.btn_portions_copy_next_click
        self.container_portions.append(self.btn_portions_copy_next)

        copy_text = "CP. Prev" if self.small_mode else "Copy to Prev"
        self.btn_portions_copy_prev = ScreenButton("btn_portions_copy_prev", copy_text, font_size, button_width + 30)
        self.btn_portions_copy_prev.set_colors(button_text_color, button_back_color)
        self.btn_portions_copy_prev.position = (self.btn_portions_copy_next.get_right() + 10, 5)
        self.btn_portions_copy_prev.click_callback = self.btn_portions_copy_prev_click
        self.container_portions.append(self.btn_portions_copy_prev)

        self.btn_portions_show.visible = not self.show_portions
        self.btn_portions_edit.visible = self.show_portions
        self.btn_portions_hide.visible = self.show_portions
        self.btn_portions_prev.visible = self.show_portions
        self.btn_portions_next.visible = self.show_portions
        self.btn_portions_del.visible = self.show_portions
        self.btn_portions_copy_next.visible = self.show_portions
        self.btn_portions_copy_prev.visible = self.show_portions

        self.undo_stack = []
        self.redo_stack = []

        self.update_current_view(True)

    def highlight_scroll_change(self, scroll):
        self.min_highlight_size = int(scroll.value)
        self.lbl_small_highlight.set_text("Highlight CCs smaller than: " + str(self.min_highlight_size))

        self.update_current_view(False)

    def update_current_view(self, resized=False):
        if self.view_mode == GTKeyFrameAnnotator.ViewModeGray:
            base_image = self.keyframe_annotations[self.selected_keyframe].grayscale_image
        elif self.view_mode == GTKeyFrameAnnotator.ViewModeBinary:
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        elif self.view_mode == GTKeyFrameAnnotator.ViewModeCombined:
            base_image = self.keyframe_annotations[self.selected_keyframe].combined_image
        else:
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image

        h, w, c = base_image.shape

        modified_image = base_image.copy()

        # show highlighted small CC (if any)
        if self.min_highlight_size > 0:
            if self.keyframe_annotations[self.selected_keyframe].binary_cc is None:
                self.keyframe_annotations[self.selected_keyframe].update_binary_image(True)

            for cc in self.keyframe_annotations[self.selected_keyframe].binary_cc:
                if cc.size < self.min_highlight_size:
                    # print(str((cc.getCenter(), cc.size, cc.min_x, cc.max_x, cc.min_y, cc.max_y)))
                    # compute highlight base radius
                    base_radius = math.sqrt(math.pow(cc.getWidth() / 2, 2.0) + math.pow(cc.getHeight() / 2, 2.0))
                    highlight_radius = int(base_radius * 3)

                    cc_cx, cc_cy = cc.getCenter()
                    cv2.circle(modified_image, (int(cc_cx), int(cc_cy)), highlight_radius, (255, 0, 0), 2)

        # show highlighted portions (if any)
        current_kf = self.keyframe_annotations[self.selected_keyframe]
        if self.show_portions and self.selected_portion[self.selected_keyframe] < len(current_kf.portions):
            portion = current_kf.portions[self.selected_portion[self.selected_keyframe]]

            cv2.rectangle(modified_image, (portion.x - 1, portion.y - 1), (portion.x + portion.w + 2, portion.y + portion.h + 2), (0, 255, 0), 2)


        # finally, resize ...
        modified_image = cv2.resize(modified_image, (int(w * self.view_scale), int(h * self.view_scale)),
                                    interpolation=cv2.INTER_NEAREST)

        self.canvas_select.height, self.canvas_select.width, _ = modified_image.shape

        # replace/update image
        self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        if resized:
            self.container_images.recalculate_size()


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
        if self.edition_mode == GTKeyFrameAnnotator.EditionModeAddPortion:
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

    def update_selected_keyframe(self, new_selected):
        if 0 <= new_selected < len(self.keyframe_annotations):
            self.selected_keyframe = new_selected
        else:
            return

        self.lbl_nav_keyframe.set_text("Key-Frame: " + str(self.selected_keyframe + 1) + " / " +
                                       str(len(self.keyframe_annotations)))

        self.lbl_portions.set_text("Portions: " + str(len(self.keyframe_annotations[self.selected_keyframe].portions)))

        time_str = TimeHelper.stampToStr(self.keyframe_annotations[self.selected_keyframe].time)
        self.lbl_nav_time.set_text(time_str)

        self.update_current_view()

    def btn_nav_keyframe_next_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe + 1)
        self.update_selected_portion()

    def btn_nav_keyframe_prev_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe - 1)
        self.update_selected_portion()

    def btn_portions_show_click(self, button):
        self.show_portions = True
        self.update_selected_portion()

    def btn_portions_hide_click(self, button):
        self.show_portions = False
        self.update_selected_portion()

    def btn_portions_add_click(self, button):
        margin = 40
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

        self.change_editor_mode(GTKeyFrameAnnotator.EditionModeAddPortion)

    def btn_portions_del_click(self, button):
        current_kf = self.keyframe_annotations[self.selected_keyframe]
        if self.show_portions and self.selected_portion[self.selected_keyframe] < len(current_kf.portions):
            prev_portions = [KeyFramePortion.Copy(portion) for portion in current_kf.portions]
            current_kf.del_portion(self.selected_portion[self.selected_keyframe])
            new_portions = [KeyFramePortion.Copy(portion) for portion in current_kf.portions]

            self.undo_stack.append({
                "operation": "portions_changed",
                "frame_idx": self.selected_keyframe,
                "prev_portions": prev_portions,
                "new_portions": new_portions,
            })

            self.update_selected_portion()

    def btn_portions_prev_click(self, button):
        if self.selected_portion[self.selected_keyframe] > 0:
            self.selected_portion[self.selected_keyframe] -= 1

        self.update_selected_portion()

    def btn_portions_next_click(self, button):
        if self.selected_portion[self.selected_keyframe] + 1 < len(self.keyframe_annotations[self.selected_keyframe].portions):
            self.selected_portion[self.selected_keyframe] += 1

        self.update_selected_portion()

    def btn_view_raw_click(self, button):
        self.view_mode = GTKeyFrameAnnotator.ViewModeRaw
        self.update_current_view()

    def btn_view_gray_click(self, button):
        self.view_mode = GTKeyFrameAnnotator.ViewModeGray
        self.update_current_view()

    def btn_view_binary_click(self, button):
        self.view_mode = GTKeyFrameAnnotator.ViewModeBinary
        self.update_current_view()

    def btn_view_combined_click(self, button):
        self.view_mode = GTKeyFrameAnnotator.ViewModeCombined
        self.update_current_view()

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False

        if to_undo["operation"] == "portions_changed":
            # restore previous portions
            affected_frame = self.keyframe_annotations[to_undo["frame_idx"]]
            affected_frame.portions = [KeyFramePortion.Copy(portion) for portion in to_undo["prev_portions"]]
            affected_frame.update_binary_image()
            success = True

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update interface ...
            self.update_selected_portion()
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

        # copy last operation
        to_redo = self.redo_stack[-1]

        success = False

        if to_redo["operation"] == "portions_changed":
            # restore new portions
            affected_frame = self.keyframe_annotations[to_redo["frame_idx"]]
            affected_frame.portions = [KeyFramePortion.Copy(portion) for portion in to_redo["new_portions"]]
            affected_frame.update_binary_image()
            success = True

        # removing ...
        if success:
            self.undo_stack.append(to_redo)
            del self.redo_stack[-1]

            # update interface ...
            self.update_selected_portion()
        else:
            print("Action could not be re-done")

    def btn_save_click(self, button):
        # first, compute the XML string for the entire set of key-frames
        xml_string = "<Annotations>\n"

        binary_path = self.output_path + "/binary"
        if not os.path.isdir(binary_path):
            os.mkdir(binary_path)

        portions_path = self.output_path + "/portions"
        if not os.path.isdir(portions_path):
            os.mkdir(portions_path)

        # for each key-frame
        xml_string += " <KeyFrames>\n"
        for keyframe in self.keyframe_annotations:
            # .... get xml string ....
            xml_string += keyframe.get_XML_string()

            # save current binarization results ...
            keyfram_portions_path = self.output_path + "/portions/frame_" + str(keyframe.idx)
            if not os.path.isdir(keyfram_portions_path):
                os.mkdir(keyfram_portions_path)

            # partial results
            for idx, portion in enumerate(keyframe.portions):
                out_bin_filename = keyfram_portions_path + "/" + str(idx) + ".png"
                cv2.imwrite(out_bin_filename, portion.binary)

            # final result combined
            out_bin_filename = binary_path + "/" + str(keyframe.idx) + ".png"
            cv2.imwrite(out_bin_filename, keyframe.binary_image)

        xml_string += " </KeyFrames>\n"
        xml_string += "</Annotations>\n"

        # store results ....
        out_xml_filename = self.output_path + "/portions.xml"
        out_xml = open(out_xml_filename, "w")
        out_xml.write(xml_string)
        out_xml.close()

        print("Annotations saved to " + out_xml_filename)

        self.undo_stack.clear()
        self.redo_stack.clear()


    def btn_exit_click(self, button):
        if len(self.undo_stack) > 0:
            # confirm before losing changes
            self.change_editor_mode(GTKeyFrameAnnotator.EditionModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")

    def change_editor_mode(self, new_mode):
        self.edition_mode = new_mode

        if self.edition_mode == GTKeyFrameAnnotator.EditionModeNavigate:

            self.container_confirm_buttons.visible = False
            self.container_state_buttons.visible = True
            self.container_view_buttons.visible = True
            self.container_nav_buttons.visible = True
            self.container_portions.visible = True

            self.canvas_select.locked = True
            self.canvas_select.elements["selection_rectangle"].visible = False

        elif self.edition_mode == GTKeyFrameAnnotator.EditionModeAddPortion:
            self.lbl_confirm_message.set_text("Select Portion to Binarize")

            self.container_confirm_buttons.visible = True
            self.container_state_buttons.visible = False
            self.container_view_buttons.visible = True
            self.container_nav_buttons.visible = False
            self.container_portions.visible = False

            self.canvas_select.locked = False
            self.canvas_select.elements["selection_rectangle"].visible = True

        elif self.edition_mode == GTKeyFrameAnnotator.EditionModeClearRegion:
            self.lbl_confirm_message.set_text("Select Portion to Clear")

            self.container_confirm_buttons.visible = True
            self.container_state_buttons.visible = False
            self.container_view_buttons.visible = True
            self.container_nav_buttons.visible = False
            self.container_portions.visible = False

            self.canvas_select.locked = False
            self.canvas_select.elements["selection_rectangle"].visible = True

        elif self.edition_mode == GTKeyFrameAnnotator.EditionModeExitConfirm:
            self.lbl_confirm_message.set_text("Unsaved Changes Will be Lost. Proceed?")

            self.container_confirm_buttons.visible = True
            self.container_state_buttons.visible = False
            self.container_view_buttons.visible = False
            self.container_nav_buttons.visible = False
            self.container_portions.visible = False

            self.canvas_select.locked = True
            self.canvas_select.elements["selection_rectangle"].visible = False

    def clear_region(self, r_x, r_y, r_w, r_h, add_undo=False):
        current_frame = self.keyframe_annotations[self.selected_keyframe]

        prev_portions = [KeyFramePortion.Copy(portion) for portion in current_frame.portions]

        pos = 0
        while pos < len(current_frame.portions):
            if current_frame.portions[pos].is_contained(r_x, r_y, r_w, r_h):
                # full containment, delete!
                del current_frame.portions[pos]
            elif current_frame.portions[pos].overlaps(r_x, r_y, r_w, r_h):
                # partial containment, clear overlap portion ...
                current_frame.portions[pos].clear_region(r_x, r_y, r_w, r_h)

                # check if portion still contains something ...
                if current_frame.portions[pos].black_pixel_count() == 0:
                    # has no more labeled pixels, remove!
                    del current_frame.portions[pos]
                else:
                    # still has content, keep and move to the next one
                    pos += 1
            else:
                # no overlap ... move next ..
                pos += 1

        new_portions = [KeyFramePortion.Copy(portion) for portion in current_frame.portions]

        if add_undo:
            self.undo_stack.append({
                "operation": "portions_changed",
                "frame_idx": self.selected_keyframe,
                "prev_portions": prev_portions,
                "new_portions": new_portions,
            })

        current_frame.update_binary_image(True)

        self.update_selected_portion()

        self.change_editor_mode(GTKeyFrameAnnotator.EditionModeNavigate)

    def btn_confirm_accept_click(self, button):
        if self.edition_mode == GTKeyFrameAnnotator.EditionModeExitConfirm:
            # exit ...
            self.return_screen = None
            print("APPLICATION FINISHED")
        elif self.edition_mode == GTKeyFrameAnnotator.EditionModeClearRegion:
            # 1) find cut
            sel_rect = self.canvas_select.elements["selection_rectangle"]
            rect_x = int(round(sel_rect.x / self.view_scale))
            rect_y = int(round(sel_rect.y / self.view_scale))
            rect_w = int(round(sel_rect.w / self.view_scale))
            rect_h = int(round(sel_rect.h / self.view_scale))

            self.clear_region(rect_x, rect_y, rect_w, rect_h, True)

        elif self.edition_mode == GTKeyFrameAnnotator.EditionModeAddPortion:
            # create binarization screen

            current_frame = self.keyframe_annotations[self.selected_keyframe]
            h, w, _ = current_frame.raw_image.shape

            # 1) find cut (correcting boundaries)
            sel_rect = self.canvas_select.elements["selection_rectangle"]
            rect_x = max(0, int(round(sel_rect.x / self.view_scale)))
            rect_y = max(0, int(round(sel_rect.y / self.view_scale)))
            rect_w = min(int(round(sel_rect.w / self.view_scale)), w - rect_x)
            rect_h = min(int(round(sel_rect.h / self.view_scale)), h - rect_y)

            # 2) create portion object
            self.editing_portion = KeyFramePortion(rect_x, rect_y, rect_w, rect_h, None)

            # 3) Get image cut
            start_x = int(max(0, rect_x))
            start_y = int(max(0, rect_y))
            end_x = int(min(w, rect_x + rect_w))
            end_y = int(min(h, rect_y + rect_h))

            image_cut = current_frame.raw_image[start_y:end_y, start_x:end_x, :].copy()

            # 4) remove objects from image cut
            mask_cut = current_frame.object_mask[start_y:end_y, start_x:end_x].copy()

            image_cut[mask_cut, 0] = 255
            image_cut[mask_cut, 1] = 255
            image_cut[mask_cut, 2] = 255

            tempo_channel = image_cut[:, :, 0].copy()
            image_cut[:, :, 0] = image_cut[:, :, 2].copy()
            image_cut[:, :, 2] = tempo_channel

            # 5) create GUI
            bin_annotator = GTBinaryAnnotator((self.width, self.height), image_cut, None, 4, self)
            bin_annotator.finished_callback = self.binarization_edition_finished
            bin_annotator.btn_view_binary_click(None)

            # 6) move to that screen
            self.return_screen = bin_annotator

    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        self.change_editor_mode(GTKeyFrameAnnotator.EditionModeNavigate)

    def binarization_edition_finished(self, accepted, binary_result):
        if accepted:
            current_frame = self.keyframe_annotations[self.selected_keyframe]
            prev_portions = [KeyFramePortion.Copy(portion) for portion in current_frame.portions]

            # add portion ..
            self.editing_portion.binary = binary_result[:, :, 0]
            current_frame.add_portion(self.editing_portion)
            self.selected_portion[self.selected_keyframe] = len(current_frame.portions) - 1

            new_portions = [KeyFramePortion.Copy(portion) for portion in current_frame.portions]

            self.undo_stack.append({
                "operation": "portions_changed",
                "frame_idx": self.selected_keyframe,
                "prev_portions": prev_portions,
                "new_portions": new_portions,
            })

        self.editing_portion = None
        self.update_selected_portion()

        # go back to navigation mode ...
        self.change_editor_mode(GTKeyFrameAnnotator.EditionModeNavigate)

    def update_selected_portion(self):
        current_kf = self.keyframe_annotations[self.selected_keyframe]
        if self.selected_portion[self.selected_keyframe] >= len(current_kf.portions):
            self.selected_portion[self.selected_keyframe] = 0

        if self.selected_portion[self.selected_keyframe] >= 0 and len(current_kf.portions) > 0:
            self.lbl_portions.set_text("Portions: " + str(self.selected_portion[self.selected_keyframe] + 1) + " / " + str(len(current_kf.portions)))
        else:
            self.lbl_portions.set_text("Portions: " + str(len(current_kf.portions)))

        #if self.show_portions:
        self.btn_portions_edit.visible = self.show_portions
        self.btn_portions_del.visible = self.show_portions
        self.btn_portions_next.visible = self.show_portions
        self.btn_portions_prev.visible = self.show_portions
        self.btn_portions_hide.visible = self.show_portions
        self.btn_portions_copy_next.visible = self.show_portions
        self.btn_portions_copy_prev.visible = self.show_portions
        self.btn_portions_show.visible = not self.show_portions

        self.update_current_view()

    def btn_portions_clear_click(self, button):
        margin = 40
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

        self.change_editor_mode(GTKeyFrameAnnotator.EditionModeClearRegion)

    def copy_frame_portion(self, source_frame_idx, source_portion_idx, target_frame_idx):
        # get portion to copy
        source_frame = self.keyframe_annotations[source_frame_idx]
        portion = source_frame.portions[source_portion_idx]

        target_frame = self.keyframe_annotations[target_frame_idx]

        source_cut = source_frame.grayscale_image[portion.y:portion.y + portion.h, portion.x:portion.x + portion.w, 0].astype(np.int32)

        h, w, _ = source_frame.grayscale_image.shape
        window_size = 10
        best_offset_y = None
        best_offset_x = None
        best_offset_score = None
        for offset_x in range(-window_size, window_size + 1):
            target_start_x = portion.x + offset_x
            target_end_x = portion.x + portion.w + offset_x

            if target_start_x < 0 or target_end_x > w:
                continue

            for offset_y in range(-window_size, window_size + 1):
                target_start_y = portion.y + offset_y
                target_end_y = portion.y + portion.h + offset_y

                if target_start_y < 0 or target_end_y > h:
                    continue

                target_cut = target_frame.grayscale_image[target_start_y:target_end_y, target_start_x:target_end_x, 0].astype(np.int32)

                distance = np.linalg.norm(source_cut - target_cut)
                if best_offset_score is None or distance < best_offset_score:
                    best_offset_score = distance
                    best_offset_y = offset_y
                    best_offset_x = offset_x

        print("Copy will be offset (" + str(best_offset_x) + ", " + str(best_offset_y) + "), offset score: " + str(best_offset_score))
        new_portion = KeyFramePortion.Copy(portion)
        new_portion.x += best_offset_x
        new_portion.y += best_offset_y

        prev_portions = [KeyFramePortion.Copy(portion) for portion in target_frame.portions]
        target_frame.add_portion(new_portion)
        new_portions = [KeyFramePortion.Copy(portion) for portion in target_frame.portions]

        self.undo_stack.append({
            "operation": "portions_changed",
            "frame_idx": target_frame_idx,
            "prev_portions": prev_portions,
            "new_portions": new_portions,
        })

    def btn_portions_edit_click(self, button):
        current_frame = self.keyframe_annotations[self.selected_keyframe]
        if self.selected_portion[self.selected_keyframe] < len(current_frame.portions):
            current_portion = current_frame[self.selected_portion[self.selected_keyframe]]
            current_binary = current_portion.binary

            # 1) find cut
            h, w, _ = current_frame.raw_image.shape
            start_x = int(max(0, current_portion.x))
            start_y = int(max(0, current_portion.y))
            end_x = int(min(w, current_portion.x + current_portion.w))
            end_y = int(min(h, current_portion.y + current_portion.h))

            image_cut = current_frame.raw_image[start_y:end_y, start_x:end_x, :].copy()

            pix_annotator = GTPixelBinaryAnnotator((self.width, self.height), image_cut, current_binary, self)
            pix_annotator.finished_callback = self.portion_edition_finished

            # 6) move to that screen
            self.return_screen = pix_annotator

    def btn_portions_copy_next_click(self, button):
        current_frame = self.keyframe_annotations[self.selected_keyframe]
        if self.selected_keyframe + 1 < len(self.keyframe_annotations) and self.selected_portion[self.selected_keyframe] < len(current_frame.portions):
            self.copy_frame_portion(self.selected_keyframe, self.selected_portion[self.selected_keyframe], self.selected_keyframe + 1)

            # move to frame, and select the new portion ....
            self.selected_portion[self.selected_keyframe + 1] = len(self.keyframe_annotations[self.selected_keyframe + 1].portions) - 1
            self.update_selected_keyframe(self.selected_keyframe + 1)
            self.update_selected_portion()

    def btn_portions_copy_prev_click(self, button):
        current_frame = self.keyframe_annotations[self.selected_keyframe]
        if self.selected_keyframe - 1 >= 0  and self.selected_portion[self.selected_keyframe] < len(current_frame.portions):
            self.copy_frame_portion(self.selected_keyframe, self.selected_portion[self.selected_keyframe], self.selected_keyframe - 1)

            # move to frame, and select the new portion ....
            self.selected_portion[self.selected_keyframe - 1] = len(self.keyframe_annotations[self.selected_keyframe - 1].portions) - 1
            self.update_selected_keyframe(self.selected_keyframe - 1)
            self.update_selected_portion()

    def portion_edition_finished(self, accepted, binary_result):
        if accepted:
            current_frame = self.keyframe_annotations[self.selected_keyframe]
            current_portion = current_frame[self.selected_portion[self.selected_keyframe]]

            current_portion.binary = binary_result

            current_frame.update_binary_image(True)

            self.update_current_view(False)
