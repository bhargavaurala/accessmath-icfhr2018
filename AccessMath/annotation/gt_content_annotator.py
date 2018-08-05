
import os
import time
import xml.etree.ElementTree as ET

import cv2

from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.annotation.video_object import VideoObject
from AccessMath.annotation.video_object_location import VideoObjectLocation
from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_canvas import ScreenCanvas
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AccessMath.interface.controls.screen_label import ScreenLabel
from AccessMath.interface.controls.screen_textbox import ScreenTextbox
from AccessMath.interface.controls.screen_textlist import ScreenTextlist
from AccessMath.interface.controls.screen_video_player import ScreenVideoPlayer


class GTContentAnnotator(Screen):
    EditGroupingTime = 5.0
    def __init__(self, size, video_files, db_name, lecture_title, output_prefix, forced_resolution=None):
        Screen.__init__(self, "Ground Truth Annotation Interface", size)

        general_background = (80, 80, 95)

        self.db_name = db_name
        self.lecture_title = lecture_title
        self.video_files = video_files

        self.output_prefix = output_prefix
        self.output_filename = output_prefix + ".xml"

        self.video_objects = {}
        self.video_segments = []
        self.video_segment_keyframes = []

        # main video player
        self.player = ScreenVideoPlayer("video_player", 960, 540)
        self.player.position = (50, 100)
        self.player.open_video_files(video_files, forced_resolution)
        self.player.frame_changed_callback = self.video_frame_change
        self.player.play()
        self.elements.append(self.player)

        print("Total Video Length: " + TimeHelper.secondsToStr(self.player.video_player.total_length))
        print("Total Video Frames: " + str(self.player.video_player.total_frames))

        # canvas used for annotations
        self.canvas = ScreenCanvas("canvas", 1040, 620)
        self.canvas.position = (10, 60)
        self.canvas.locked = True
        self.canvas.object_edited_callback = self.canvas_object_edited
        self.canvas.object_selected_callback = self.canvas_selection_changed
        self.elements.append(self.canvas)

        self.last_video_frame = None
        self.last_video_time = None

        # add elements....
        # TITLE
        label_title = ScreenLabel("title", "ACCESS MATH - Video Annotation Tool", 28)
        label_title.background = general_background
        label_title.position = (int((self.width - label_title.width) / 2), 20)
        label_title.set_color((255, 255, 255))
        self.elements.append(label_title)

        # EXIT BUTTON
        exit_button = ScreenButton("exit_button", "EXIT", 16, 70, 0)
        exit_button.set_colors((192, 255, 128), (64, 64, 64))
        exit_button.position = (self.width - exit_button.width - 15, self.height - exit_button.height - 15)
        exit_button.click_callback = self.close_click
        self.elements.append(exit_button)

        # video controllers
        self.container_video_controls = ScreenContainer("container_video_controls", (1050, 200), general_background)
        self.container_video_controls.position = (5, self.canvas.get_bottom() + 5)
        self.elements.append(self.container_video_controls)

        step_1 = self.player.video_player.total_frames / 100
        self.position_scroll = ScreenHorizontalScroll("video_position", 0, self.player.video_player.total_frames -1, 0, step_1)
        self.position_scroll.position = (5, 5)
        self.position_scroll.width = 1040
        self.position_scroll.scroll_callback = self.main_scroll_change
        self.container_video_controls.append(self.position_scroll)

        # Frame count
        self.label_frame_count = ScreenLabel("frame_count", "Frame Count: " + str(int(self.player.video_player.total_frames)), 18)
        self.label_frame_count.position = (15, self.position_scroll.get_bottom() + 10)
        self.label_frame_count.set_color((255, 255, 255))
        self.label_frame_count.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_frame_count)

        # Current Frame
        self.label_frame_current = ScreenLabel("frame_current", "Current Frame: 0", 18)
        self.label_frame_current.position = (175, int(self.label_frame_count.get_top()))
        self.label_frame_current.set_color((255, 255, 255))
        self.label_frame_current.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_frame_current)

        # Current Time
        self.label_time_current = ScreenLabel("time_current", "Current Time: 0", 18)
        self.label_time_current.position = (175, int(self.label_frame_current.get_bottom() + 15))
        self.label_time_current.set_color((255, 255, 255))
        self.label_time_current.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_time_current)

        # player speed
        self.label_player_speed = ScreenLabel("label_player_speed", "Speed: 100%", 18)
        self.label_player_speed.position = (475, int(self.label_frame_count.get_top()))
        self.label_player_speed.set_color((255, 255, 255))
        self.label_player_speed.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_player_speed)

        # Player speed buttons
        dec_speed = ScreenButton("dec_speed", "0.5x", 16, 70, 0)
        dec_speed.set_colors((192, 255, 128), (64, 64, 64))
        dec_speed.position = (self.label_player_speed.get_left() - dec_speed.width - 15, self.label_player_speed.get_top())
        dec_speed.click_callback = self.btn_dec_speed_click
        self.container_video_controls.append(dec_speed)

        inc_speed = ScreenButton("inc_speed", "2.0x", 16, 70, 0)
        inc_speed.set_colors((192, 255, 128), (64, 64, 64))
        inc_speed.position = (self.label_player_speed.get_right() + 15, self.label_player_speed.get_top())
        inc_speed.click_callback = self.btn_inc_speed_click
        self.container_video_controls.append(inc_speed)

        # Precision buttons ....
        v_pos = self.label_time_current.get_bottom() + 15
        btn_w = 70
        for idx, value in enumerate([-1000, -100, -10, -1]):
            prec_button = ScreenButton("prec_button_m_" + str(idx), str(value), 16, btn_w, 0)
            prec_button.set_colors((192, 255, 128), (64, 64, 64))
            prec_button.position = (15 + idx * (btn_w + 15) , v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)

        self.button_pause = ScreenButton("btn_pause", "Pause", 16, 70, 0)
        self.button_pause.set_colors((192, 255, 128), (64, 64, 64))
        self.button_pause.position = (15 + 4 * (btn_w + 15), v_pos)
        self.button_pause.click_callback = self.btn_pause_click
        self.container_video_controls.append(self.button_pause)

        self.button_play = ScreenButton("btn_play", "Play", 16, 70, 0)
        self.button_play.set_colors((192, 255, 128), (64, 64, 64))
        self.button_play.position = (15 + 4 * (btn_w + 15), v_pos)
        self.button_play.click_callback = self.btn_play_click
        self.button_play.visible = False
        self.container_video_controls.append(self.button_play)

        for idx, value in enumerate([1, 10, 100, 1000]):
            prec_button = ScreenButton("prec_button_p_" + str(idx), str(value), 16, btn_w, 0)
            prec_button.set_colors((192, 255, 128), (64, 64, 64))
            prec_button.position = (15 + (5 + idx) * (btn_w + 15) , v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)

        self.elements.back_color = general_background

        # Container 1: Object selector and controls ...
        self.container_object_options = ScreenContainer("container_object_options", (425, 390), general_background)
        self.container_object_options.position = (self.canvas.get_right() + 5, self.canvas.get_top())

        # ... Object selector ...
        self.object_selector = ScreenTextlist("object_selector", (295, 340), 21, (40, 40, 48), (255, 255, 255),
                                              (190, 190, 128), (0, 0, 0))
        self.object_selector.position = (10, 40)
        self.object_selector.selected_value_change_callback = self.object_selector_option_changed
        self.container_object_options.append(self.object_selector)
        # ... label ...
        label_object_selector = ScreenLabel("label_object_selector", "Video Objects", 26)
        label_object_selector.background = general_background
        label_object_selector.position = (self.object_selector.get_center_x() - label_object_selector.width / 2, 5)
        label_object_selector.set_color((255, 255, 255))
        self.container_object_options.append(label_object_selector)

        # ... object buttons ....
        # ...... add ....
        btn_object_add = ScreenButton("btn_object_add", "Add", 22, 100)
        btn_object_add.set_colors((192, 255, 128), (64, 64, 64))
        btn_object_add.position = (self.container_object_options.width - btn_object_add.width - 5, self.object_selector.get_top())
        btn_object_add.click_callback = self.btn_object_add
        self.container_object_options.append(btn_object_add)
        # ...... rename ....
        btn_object_rename = ScreenButton("btn_object_rename", "Rename", 22, 100)
        btn_object_rename.set_colors((192, 255, 128), (64, 64, 64))
        btn_object_rename.position = (self.container_object_options.width - btn_object_rename.width - 5, btn_object_add.get_bottom() + 15)
        btn_object_rename.click_callback = self.btn_object_rename
        self.container_object_options.append(btn_object_rename)
        # ...... remove ....
        btn_object_remove = ScreenButton("btn_object_remove", "Remove", 22, 100)
        btn_object_remove.set_colors((192, 255, 128), (64, 64, 64))
        btn_object_remove.position = (self.container_object_options.width - btn_object_remove.width - 5, btn_object_rename.get_bottom() + 15)
        btn_object_remove.click_callback = self.btn_object_remove
        self.container_object_options.append(btn_object_remove)

        self.elements.append(self.container_object_options)

        # ... key-frame buttons ...
        self.container_keyframe_options = ScreenContainer("container_keyframe_options", (425, 100), general_background)
        self.container_keyframe_options.position = (self.canvas.get_right() + 5, self.container_object_options.get_bottom() + 5)
        self.container_keyframe_options.visible = False

        # lbl_keyframe_title
        self.lbl_keyframe_title = ScreenLabel("lbl_keyframe_title", "Object Key-frames: ", 21, 415, 1)
        self.lbl_keyframe_title.set_color((255,255, 255))
        self.lbl_keyframe_title.set_background(general_background)
        self.lbl_keyframe_title.position = (5, 5)
        self.container_keyframe_options.append(self.lbl_keyframe_title)

        # btn_keyframe_invisible
        self.btn_keyframe_invisible = ScreenButton("btn_keyframe_invisible", "Hide", 22, 75)
        self.btn_keyframe_invisible.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_invisible.position = (25, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_invisible.click_callback = self.btn_keyframe_invisible_click
        self.container_keyframe_options.append(self.btn_keyframe_invisible)

        # btn_keyframe_visible
        self.btn_keyframe_visible = ScreenButton("btn_keyframe_visible", "Show", 22, 75)
        self.btn_keyframe_visible.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_visible.position = (25, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_visible.click_callback = self.btn_keyframe_visible_click
        self.container_keyframe_options.append(self.btn_keyframe_visible)

        # btn_keyframe_prev
        self.btn_keyframe_prev = ScreenButton("btn_keyframe_prev", "Prev", 22, 75)
        self.btn_keyframe_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_prev.position = (125, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_prev.click_callback = self.btn_jump_frame_click
        self.container_keyframe_options.append(self.btn_keyframe_prev)

        # btn_keyframe_next
        self.btn_keyframe_next = ScreenButton("btn_keyframe_next", "Next", 22, 75)
        self.btn_keyframe_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_next.position = (225, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_next.click_callback = self.btn_jump_frame_click
        self.container_keyframe_options.append(self.btn_keyframe_next)

        # btn_keyframe_add
        self.btn_keyframe_add = ScreenButton("btn_keyframe_add", "Add", 22, 75)
        self.btn_keyframe_add.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_add.position = (325, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_add.click_callback = self.btn_keyframe_add_click
        self.container_keyframe_options.append(self.btn_keyframe_add)

        # btn_keyframe_del
        self.btn_keyframe_del = ScreenButton("btn_keyframe_del", "Del", 22, 75)
        self.btn_keyframe_del.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_del.position = (325, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_del.click_callback = self.btn_keyframe_del_click
        self.container_keyframe_options.append(self.btn_keyframe_del)

        # lbl_keyframe_prev
        self.lbl_keyframe_prev = ScreenLabel("lbl_keyframe_prev", "[0]", 21, 75)
        self.lbl_keyframe_prev.position = (self.btn_keyframe_prev.get_left(), self.btn_keyframe_prev.get_bottom() + 10)
        self.lbl_keyframe_prev.set_color((255, 255, 255))
        self.lbl_keyframe_prev.set_background(general_background)
        self.container_keyframe_options.append(self.lbl_keyframe_prev)

        # lbl_keyframe_next
        self.lbl_keyframe_next = ScreenLabel("lbl_keyframe_next", "[0]", 21, 75)
        self.lbl_keyframe_next.position = (self.btn_keyframe_next.get_left(), self.btn_keyframe_next.get_bottom() + 10)
        self.lbl_keyframe_next.set_color((255, 255, 255))
        self.lbl_keyframe_next.set_background(general_background)
        self.container_keyframe_options.append(self.lbl_keyframe_next)

        self.elements.append(self.container_keyframe_options)

        # ... Video Segments buttons ...
        self.container_vid_seg_options = ScreenContainer("container_vid_seg_options", (425, 100), general_background)
        self.container_vid_seg_options.position = (self.canvas.get_right() + 5, self.container_keyframe_options.get_bottom() + 5)
        self.container_vid_seg_options.visible = True

        # lbl_vid_seg_title
        self.lbl_vid_seg_title = ScreenLabel("lbl_vid_seg_title", "Video Segments: ", 21, 415, 1)
        self.lbl_vid_seg_title.set_color((255,255, 255))
        self.lbl_vid_seg_title.set_background(general_background)
        self.lbl_vid_seg_title.position = (5, 5)
        self.container_vid_seg_options.append(self.lbl_vid_seg_title)

        # btn_vid_seg_prev
        self.btn_vid_seg_prev = ScreenButton("btn_vid_seg_prev", "Prev", 22, 110)
        self.btn_vid_seg_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_prev.position = (20, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_prev.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_options.append(self.btn_vid_seg_prev)

        # btn_vid_seg_next
        self.btn_vid_seg_next = ScreenButton("btn_vid_seg_next", "Next", 22, 110)
        self.btn_vid_seg_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_next.position = (157, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_next.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_options.append(self.btn_vid_seg_next)

        # btn_vid_seg_split
        self.btn_vid_seg_split = ScreenButton("btn_vid_seg_split", "Split", 22, 110)
        self.btn_vid_seg_split.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_split.position = (295, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_split.click_callback = self.btn_vid_seg_split_click
        self.container_vid_seg_options.append(self.btn_vid_seg_split)

        # btn_vid_seg_merge
        self.btn_vid_seg_merge = ScreenButton("btn_vid_seg_merge", "Merge", 22, 110)
        self.btn_vid_seg_merge.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_merge.position = (295, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_merge.click_callback = self.btn_vid_seg_merge_click
        self.container_vid_seg_options.append(self.btn_vid_seg_merge)

        # lbl_vid_seg_prev
        self.lbl_vid_seg_prev = ScreenLabel("lbl_vid_seg_prev", "[0]", 21, 110)
        self.lbl_vid_seg_prev.position = (self.btn_vid_seg_prev.get_left(), self.btn_vid_seg_prev.get_bottom() + 10)
        self.lbl_vid_seg_prev.set_color((255, 255, 255))
        self.lbl_vid_seg_prev.set_background(general_background)
        self.container_vid_seg_options.append(self.lbl_vid_seg_prev)

        # lbl_vid_seg_next
        self.lbl_vid_seg_next = ScreenLabel("lbl_vid_seg_next", "[0]", 21, 110)
        self.lbl_vid_seg_next.position = (self.btn_vid_seg_next.get_left(), self.btn_vid_seg_next.get_bottom() + 10)
        self.lbl_vid_seg_next.set_color((255, 255, 255))
        self.lbl_vid_seg_next.set_background(general_background)
        self.container_vid_seg_options.append(self.lbl_vid_seg_next)

        self.elements.append(self.container_vid_seg_options)

        # ... Video Segments buttons ...
        self.container_vid_seg_keyframe_options = ScreenContainer("container_vid_seg_keyframe_options", (425, 100), general_background)
        self.container_vid_seg_keyframe_options.position = (self.canvas.get_right() + 5, self.container_vid_seg_options.get_bottom() + 5)
        self.container_vid_seg_keyframe_options.visible = True

        # lbl_vid_seg_keyframe_title
        self.lbl_vid_seg_keyframe_title = ScreenLabel("lbl_vid_seg_keyframe_title", "Segment Keyframes: ", 21, 415, 1)
        self.lbl_vid_seg_keyframe_title.set_color((255,255, 255))
        self.lbl_vid_seg_keyframe_title.set_background(general_background)
        self.lbl_vid_seg_keyframe_title.position = (5, 5)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_title)

        # btn_vid_seg_keyframe_prev
        self.btn_vid_seg_keyframe_prev = ScreenButton("btn_vid_seg_keyframe_prev", "Prev", 22, 110)
        self.btn_vid_seg_keyframe_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_prev.position = (20, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_prev.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_prev)

        # btn_vid_seg_keyframe_next
        self.btn_vid_seg_keyframe_next = ScreenButton("btn_vid_seg_keyframe_next", "Next", 22, 110)
        self.btn_vid_seg_keyframe_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_next.position = (157, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_next.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_next)

        # btn_vid_seg_keyframe_add
        self.btn_vid_seg_keyframe_add = ScreenButton("btn_vid_seg_keyframe_split", "Add", 22, 110)
        self.btn_vid_seg_keyframe_add.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_add.position = (295, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_add.click_callback = self.btn_vid_seg_keyframe_add_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_add)

        # btn_vid_seg_keyframe_del
        self.btn_vid_seg_keyframe_del = ScreenButton("btn_vid_seg_keyframe_del", "Del", 22, 110)
        self.btn_vid_seg_keyframe_del.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_del.position = (295, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_del.click_callback = self.btn_vid_seg_keyframe_del_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_del)

        # lbl_vid_seg_keyframe_prev
        self.lbl_vid_seg_keyframe_prev = ScreenLabel("lbl_vid_seg_keyframe_prev", "[0]", 21, 110)
        self.lbl_vid_seg_keyframe_prev.position = (self.btn_vid_seg_keyframe_prev.get_left(), self.btn_vid_seg_keyframe_prev.get_bottom() + 10)
        self.lbl_vid_seg_keyframe_prev.set_color((255, 255, 255))
        self.lbl_vid_seg_keyframe_prev.set_background(general_background)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_prev)

        # lbl_vid_seg_keyframe_next
        self.lbl_vid_seg_keyframe_next = ScreenLabel("lbl_vid_seg_keyframe_next", "[0]", 21, 110)
        self.lbl_vid_seg_keyframe_next.position = (self.btn_vid_seg_keyframe_next.get_left(), self.btn_vid_seg_keyframe_next.get_bottom() + 10)
        self.lbl_vid_seg_keyframe_next.set_color((255, 255, 255))
        self.lbl_vid_seg_keyframe_next.set_background(general_background)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_next)

        self.elements.append(self.container_vid_seg_keyframe_options)

        # Container for text input ....
        self.container_text_input = ScreenContainer("container_text_input", (425, 530), general_background)
        self.container_text_input.position = (self.canvas.get_right() + 5, self.canvas.get_top())

        # ...text box ...
        self.txt_object_name = ScreenTextbox("txt_object_name", "", 25, 280)
        self.txt_object_name.position = (int((self.container_text_input.width  - self.txt_object_name.width) / 2), 80)
        self.txt_object_name.set_colors((255, 255, 255), (40, 40, 48))
        self.container_text_input.append(self.txt_object_name)
        # ...title for text box ...
        self.label_txt_object_name = ScreenLabel("label_txt_object_name", "Object Name", 21, max_width=280)
        self.label_txt_object_name.position = (self.txt_object_name.get_left(), 60)
        self.label_txt_object_name.set_color((255, 255, 255))
        self.label_txt_object_name.set_background(general_background)
        self.container_text_input.append(self.label_txt_object_name)
        # ... accept button ...
        btn_text_operation_accept = ScreenButton("btn_text_operation_accept", "Accept", 22, 100)
        btn_text_operation_accept.set_colors((192, 255, 128), (64, 64, 64))
        btn_text_operation_accept.position = (self.txt_object_name.get_center_x() - 120, self.txt_object_name.get_bottom() + 30)
        btn_text_operation_accept.click_callback = self.btn_text_operation_accept_click
        self.container_text_input.append(btn_text_operation_accept)
        # ... cancel button ...
        btn_text_operation_cancel = ScreenButton("btn_text_operation_cancel", "Cancel", 22, 100)
        btn_text_operation_cancel.set_colors((192, 255, 128), (64, 64, 64))
        btn_text_operation_cancel.position = (self.txt_object_name.get_center_x() + 20, self.txt_object_name.get_bottom() + 30)
        btn_text_operation_cancel.click_callback = self.btn_text_operation_cancel_click
        self.container_text_input.append(btn_text_operation_cancel)
        # ...error display label...
        self.label_txt_object_error = ScreenLabel("label_txt_object_error", "Displaying Error Messages", 21, max_width=280)
        self.label_txt_object_error.position = (self.txt_object_name.get_left(), btn_text_operation_accept.get_bottom() + 15)
        self.label_txt_object_error.set_color((255, 64, 64))
        self.label_txt_object_error.set_background(general_background)
        self.container_text_input.append(self.label_txt_object_error)

        self.container_text_input.visible = False
        self.elements.append(self.container_text_input)

        # SAVE BUTTON
        save_button = ScreenButton("save_button", "SAVE", 16, 100, 0)
        save_button.set_colors((192, 255, 128), (64, 64, 64))
        save_button.position = (self.width - save_button.width - 20, self.container_vid_seg_keyframe_options.get_bottom() + 20)
        save_button.click_callback = self.save_data_click
        self.elements.append(save_button)

        # EXPORT Buttton
        export_button = ScreenButton("export_button", "EXPORT SEGMENTS", 16, 150, 0)
        export_button.set_colors((192, 255, 128), (64, 64, 64))
        export_button.position = (save_button.get_left() - export_button.width - 20, self.container_vid_seg_keyframe_options.get_bottom() + 20)
        export_button.click_callback = self.btn_export_segments_click
        self.elements.append(export_button)

        # REDO Button
        redo_button = ScreenButton("redo_button", "REDO", 16, 100, 0)
        redo_button.set_colors((192, 255, 128), (64, 64, 64))
        redo_button.position = (export_button.get_left() - redo_button.width - 50, self.container_vid_seg_keyframe_options.get_bottom() + 20)
        redo_button.click_callback = self.btn_redo_click
        self.elements.append(redo_button)

        # UNDO Button
        undo_button = ScreenButton("undo_button", "UNDO", 16, 100, 0)
        undo_button.set_colors((192, 255, 128), (64, 64, 64))
        undo_button.position = (redo_button.get_left() - redo_button.width - 20, self.container_vid_seg_keyframe_options.get_bottom() + 20)
        undo_button.click_callback = self.btn_undo_click
        self.elements.append(undo_button)

        # 0 - Nothing
        # 1 - Adding new Object
        # 2 - Renaming Object
        # 3 - Confirm Deleting Object
        # 4 - Confirm Exit without saving
        self.text_operation = 0
        self.changes_saved = True
        self.undo_stack = []
        self.redo_stack = []

        if os.path.exists(self.output_filename):
            print("Saved file exists. Loading ...")

            self.load_saved_data()

        self.update_video_segment_buttons()

    def show_xml_metadata(self, root):
        # Loading and showing metadata
        database_name = root.find(VideoObject.XMLNamespace + 'Database').text
        lecture_title = root.find(VideoObject.XMLNamespace + 'Lecture').text
        output_file = root.find(VideoObject.XMLNamespace + 'Filename').text

        tempo_videos = []
        file_videos = root.find(VideoObject.XMLNamespace + 'VideoFiles')
        for file_video in file_videos.findall(VideoObject.XMLNamespace + 'VideoFile'):
            tempo_videos.append(file_video.text)

        print("Loading data:")
        print("- Database: " + str(database_name))
        print("- Lecture: " + str(lecture_title))
        print("- Output: " + str(output_file))
        print("- Videos: ")
        for file_video in tempo_videos:
            print("\t" + file_video)



    def load_saved_data(self):
        tree = ET.parse(self.output_filename)
        root = tree.getroot()

        # Show metada just for validation purposes
        self.show_xml_metadata(root)

        namespace = VideoObject.XMLNamespace

        # load video objects
        xml_video_objects_root = root.find(namespace + 'VideoObjects')
        xml_video_objects = xml_video_objects_root.findall(namespace + 'VideoObject')
        for xml_video_object in xml_video_objects:
            # load logical object ...
            video_object = VideoObject.fromXML(xml_video_object)

            print(" -> Loading object: " + video_object.name + " (" + str(len(video_object.locations)) + " Key-frames)")

            # logical object
            self.video_objects[video_object.id] = video_object

            # add to the interface
            # ... txtlist
            self.object_selector.add_option(video_object.id, video_object.name)

            # ... canvas
            self.canvas.add_element(video_object.id, 0, 0, 0, 0)
            self.canvas.elements[video_object.id].visible = False

        # load video segments ...
        xml_video_segments_root = root.find(namespace + "VideoSegments")
        xml_video_segment_objects = xml_video_segments_root.findall(namespace + "VideoSegment")
        tempo_split_points = []
        for xml_video_segment_object in xml_video_segment_objects:
            split_point = int(xml_video_segment_object.find(VideoObject.XMLNamespace + 'Start').text)
            tempo_split_points.append(split_point)

        tempo_split_points = sorted(tempo_split_points)
        if 0 in tempo_split_points:
            tempo_split_points.remove(0)

        self.video_segments = tempo_split_points

        # load key-frames ...
        xml_video_keyframes_root = root.find(namespace + "VideoKeyFrames")
        xml_video_keyframes_objects = xml_video_keyframes_root.findall(namespace + "VideoKeyFrame")
        tempo_keyframes = []
        for xml_video_keyframe_object in xml_video_keyframes_objects:
            frame_idx = int(xml_video_keyframe_object.find(namespace + "Index").text)
            tempo_keyframes.append(frame_idx)

        tempo_keyframes = sorted(tempo_keyframes)
        self.video_segment_keyframes = tempo_keyframes


    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False
        if to_undo["operation"] == "object_added":
            # inverse of adding is removing ...
            success = self.remove_object(to_undo["id"])

        elif to_undo["operation"] == "object_renamed":
            # inverse of renaming, is going back to old name  ...
            success = self.rename_object(to_undo["new_id"], to_undo["old_id"], to_undo["old_display"])

        elif to_undo["operation"] == "object_removed":
            # inverse of removing, adding back ..
            old_object = to_undo["object_ref"]
            first_loc = old_object.locations[0]
            success = self.add_object(to_undo["id"], to_undo["display"], 0, 0, 0, 0, 0, 0)
            if success:
                new_object = self.video_objects[to_undo["id"]]
                # overwrite locations
                new_object.locations = old_object.locations
                # update object reference
                to_undo["object_ref"] = new_object

        elif to_undo["operation"] == "keyframe_added":
            success = self.video_objects[to_undo["object_id"]].del_location_at(to_undo["new_location"].frame)

        elif to_undo["operation"] == "keyframe_edited" or to_undo["operation"] == "keyframe_deleted":
            # return key-frame to previous state (either modify or add back)
            pre_loc = to_undo["old_location"]
            self.video_objects[to_undo["object_id"]].set_location_at(pre_loc.frame, pre_loc.abs_time, pre_loc.visible,
                                                                     pre_loc.x, pre_loc.y, pre_loc.w, pre_loc.h)
            success = True

        elif to_undo["operation"] == "vid_seg_split":
            # to undo split ... merge
            success = self.segment_merge(to_undo["split_point"], False)
        elif to_undo["operation"] == "vid_seg_merge":
            # to undo merge ... split
            success = self.segment_split(to_undo["split_point"], False)
        elif to_undo["operation"] == "vid_seg_keyframe_add":
            # del key-frame
            success = self.segment_keyframe_del(to_undo["frame_index"], False)
        elif to_undo["operation"] == "vid_seg_keyframe_del":
            # add key-frame
            success = self.segment_keyframe_add(to_undo["frame_index"], False)

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update canvas ...
            self.update_canvas_objects()

            # update key-frame information
            self.update_keyframe_buttons()
            self.update_video_segment_buttons()
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

        # copy last operation
        to_redo = self.redo_stack[-1]

        success = False
        if to_redo["operation"] == "object_added":
            loc = to_redo["location"]
            success = self.add_object(to_redo["id"], to_redo["name"], loc.frame, loc.abs_time, loc.x, loc.y, loc.w, loc.h)

        elif to_redo["operation"] == "object_renamed":
            success = self.rename_object(to_redo["old_id"], to_redo["new_id"], to_redo["new_display"])

        elif to_redo["operation"] == "object_removed":
            success = self.remove_object(to_redo["id"])

        elif to_redo["operation"] == "keyframe_added" or to_redo["operation"] == "keyframe_edited":
            add_loc = to_redo["new_location"]
            self.video_objects[to_redo["object_id"]].set_location_at(add_loc.frame, add_loc.abs_time, add_loc.visible,
                                                                     add_loc.x, add_loc.y, add_loc.w, add_loc.h)
            success = True

        elif to_redo["operation"] == "keyframe_deleted":
            # return key-frame to previous state (either modify or add back)
            success = self.video_objects[to_redo["object_id"]].del_location_at(to_redo["old_location"].frame)

        elif to_redo["operation"] == "vid_seg_split":
            success = self.segment_split(to_redo["split_point"], False)
        elif to_redo["operation"] == "vid_seg_merge":
            success = self.segment_merge(to_redo["split_point"], False)
        elif to_redo["operation"] == "vid_seg_keyframe_add":
            success = self.segment_keyframe_add(to_redo["frame_index"], False)
        elif to_redo["operation"] == "vid_seg_keyframe_del":
            success = self.segment_keyframe_del(to_redo["frame_index"], False)

        if success:
            self.undo_stack.append(to_redo)
            # removing last operation
            del self.redo_stack[-1]

            # update canvas ...
            self.update_canvas_objects()

            # update key-frame information
            self.update_keyframe_buttons()
            self.update_video_segment_buttons()
        else:
            print("Action could not be re-done!")

    def add_object(self, id, name, frame, abs_time, x, y, w, h):
        if name in self.video_objects:
            print("The Object named <" + id + "> already exists!")
            return False

        # add to objects ...
        self.video_objects[id] = VideoObject(id, name)
        self.video_objects[id].set_location_at(frame, abs_time, True, x, y, w, h)

        # add to canvas ....
        self.canvas.add_element(id, x, y, w, h)

        # add to text list
        self.object_selector.add_option(id, name)

        self.changes_saved = False

        return True

    def rename_object(self, old_id, new_id, new_name):
        if old_id != new_id:
            # name changed!, verify ...
            if new_id in self.video_objects:
                print("Object name already in use")
                return False

            # valid name change, call rename operations
            # ... canvas ...
            self.canvas.rename_element(old_id, new_id)
            # ... object selector ...
            self.object_selector.rename_option(old_id, new_id, new_name)
            # ... reference to object ....
            # .... copy ref ...
            self.video_objects[new_id] = self.video_objects[old_id]
            # .... remove old ref ...
            del self.video_objects[old_id]
            # .... change object name
            self.video_objects[new_id].id = new_id
            self.video_objects[new_id].name = new_name

            self.changes_saved = False

        return True

    def remove_object(self, object_name):
        if object_name not in self.video_objects:
            print("Cannot remove object")
            return False

        # ... remove from canvas
        self.canvas.remove_element(object_name)
        # ... remove from object selector
        self.object_selector.remove_option(object_name)
        # ... remove from video objects
        del self.video_objects[object_name]

        self.changes_saved = False

        return True

    def main_scroll_change(self, scroll):
        self.player.set_player_frame(int(scroll.value), True)

    def update_canvas_objects(self):
        # update canvas objects ....
        for object_name in self.video_objects:
            loc = self.video_objects[object_name].get_location_at(self.last_video_frame, False)

            if loc is None:
                self.canvas.update_element(object_name, 0,0,0,0, False)
            else:
                self.canvas.update_element(object_name, loc.x, loc.y, loc.w, loc.h, loc.visible)

    def video_frame_change(self, next_frame, next_abs_time):
        # update the scroll bar
        self.position_scroll.value = next_frame

        self.last_video_frame = next_frame
        self.last_video_time = next_abs_time

        self.label_frame_current.set_text("Current frame: " + str(next_frame))
        self.label_time_current.set_text("Current time: " + TimeHelper.stampToStr(next_abs_time))

        # update canvas ...
        self.update_canvas_objects()

        # update key-frame information
        self.update_keyframe_buttons()

        # udpate segment info ...
        self.update_video_segment_buttons()


    def handle_events(self, events):
        #handle other events
        return super(GTContentAnnotator, self).handle_events(events)


    def render(self, background):
        # draw other controls..
        super(GTContentAnnotator, self).render(background)

    def close_click(self, button):
        if self.changes_saved:
            self.return_screen = None
            print("APPLICATION FINISHED")
        else:
            self.text_operation = 3
            print("Warning: Last changes have not been saved")
            # set exit confirm mode
            self.prepare_confirm_input_mode(4, None, "Exit without saving?")

    def project_object_location(self, loc):
        assert isinstance(loc, VideoObjectLocation)

        off_x = self.player.render_location[0] - self.canvas.position[0]
        off_y = self.player.render_location[1] - self.canvas.position[1]
        # note that these values should be the same (if aspect ratio is kept)
        scale_x = self.player.video_player.width / self.player.render_width
        scale_y = self.player.video_player.height / self.player.render_height

        proj_x = (loc.x - off_x) * scale_x
        proj_y = (loc.y - off_y) * scale_y
        proj_w = loc.w * scale_x
        proj_h = loc.h * scale_y

        return VideoObjectLocation(loc.visible, loc.frame, loc.abs_time, proj_x, proj_y, proj_w, proj_h)

    def generate_video_segments_xml(self):
        tempo_segments = [0] + self.video_segments + [self.player.video_player.total_frames]
        xml_string = "  <VideoSegments>\n"
        for idx in range(len(self.video_segments) + 1):
            xml_string += "    <VideoSegment>\n"
            xml_string += "        <Start>" + str(tempo_segments[idx]) + "</Start>\n"
            xml_string += "        <End>" + str(tempo_segments[idx + 1]) + "</End>\n"
            xml_string += "    </VideoSegment>\n"
        xml_string += "  </VideoSegments>\n"

        return xml_string

    def generate_keyframes_xml(self, include_objects, keyframe_times=None):
        xml_string = "  <VideoKeyFrames>\n"
        for idx, frame_idx in enumerate(self.video_segment_keyframes):
            xml_string += "    <VideoKeyFrame>\n"
            xml_string += "       <Index>" + str(frame_idx) + "</Index>\n"

            if keyframe_times is not None:
                xml_string += "       <AbsTime>" + str(keyframe_times[idx]) + "</AbsTime>\n"

            if include_objects:
                xml_string += "       <VideoObjects>\n"

                # for each object ....
                for object_name in self.video_objects:
                    # get location of object at current key-frame
                    loc = self.video_objects[object_name].get_location_at(frame_idx, False)

                    # only add if object is visible at current key-frame
                    if loc is not None and loc.visible:
                        proj_loc = self.project_object_location(loc)

                        object_xml = "          <VideoObject>\n"
                        object_xml += "              <Name>" + object_name + "</Name>\n"
                        object_xml += "              <X>" + str(proj_loc.x) + "</X>\n"
                        object_xml += "              <Y>" + str(proj_loc.y) + "</Y>\n"
                        object_xml += "              <W>" + str(proj_loc.w) + "</W>\n"
                        object_xml += "              <H>" + str(proj_loc.h) + "</H>\n"
                        object_xml += "          </VideoObject>\n"

                        xml_string += object_xml

                xml_string += "       </VideoObjects>\n"

            xml_string += "    </VideoKeyFrame>\n"

        xml_string += "  </VideoKeyFrames>\n"

        return xml_string


    def generate_data_xml(self):
        xml_string = "<Annotations>\n"

        # general meta-data
        xml_string += self.generate_metadata_header_xml()

        # add ViewPort coordinates info ...
        xml_string += "  <DrawingInfo>\n"
        xml_string += "     <Canvas>\n"
        xml_string += "         <X>" + str(self.canvas.position[0]) + "</X>\n"
        xml_string += "         <Y>" + str(self.canvas.position[1]) + "</Y>\n"
        xml_string += "         <W>" + str(self.canvas.width) + "</W>\n"
        xml_string += "         <H>" + str(self.canvas.height) + "</H>\n"
        xml_string += "     </Canvas>\n"
        xml_string += "     <Player>\n"
        xml_string += "         <ControlArea>\n"
        xml_string += "             <X>" + str(self.player.position[0]) + "</X>\n"
        xml_string += "             <Y>" + str(self.player.position[1]) + "</Y>\n"
        xml_string += "             <W>" + str(self.player.width) + "</W>\n"
        xml_string += "             <H>" + str(self.player.height) + "</H>\n"
        xml_string += "         </ControlArea>\n"
        xml_string += "         <RenderArea>\n"
        xml_string += "             <X>" + str(self.player.render_location[0]) + "</X>\n"
        xml_string += "             <Y>" + str(self.player.render_location[1]) + "</Y>\n"
        xml_string += "             <W>" + str(self.player.render_width) + "</W>\n"
        xml_string += "             <H>" + str(self.player.render_height) + "</H>\n"
        xml_string += "         </RenderArea>\n"
        xml_string += "     </Player>\n"
        xml_string += "  </DrawingInfo>\n"


        xml_string += "  <VideoObjects>\n"
        for name in self.video_objects:
            xml_string += self.video_objects[name].toXML()
        xml_string += "  </VideoObjects>\n"

        xml_string += self.generate_video_segments_xml()

        # save key-frames without object info (full object info already saved)
        xml_string += self.generate_keyframes_xml(False)

        xml_string += "</Annotations>\n"

        return xml_string

    def save_data_click(self, button):
        xml_data = self.generate_data_xml()

        out_file = open(self.output_filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("Saved to: " + self.output_filename)
        self.changes_saved = True
        # Free the queues
        self.undo_stack.clear()
        self.redo_stack.clear()

    def btn_dec_speed_click(self, button):
        self.player.decrease_speed()

        self.label_player_speed.set_text("Speed: " + str(self.player.video_player.play_speed * 100.0) + "%")

    def btn_inc_speed_click(self, button):
        self.player.increase_speed()

        self.label_player_speed.set_text("Speed: " + str(self.player.video_player.play_speed * 100.0) + "%")

    def btn_change_frame(self, button):
        new_abs_frame = self.player.video_player.last_frame_idx + button.tag
        self.player.set_player_frame(new_abs_frame, True)

    def btn_pause_click(self, button):
        self.player.pause()
        self.canvas.locked = False

        self.button_play.visible = True
        self.button_pause.visible = False

    def btn_play_click(self, button):
        self.player.play()
        self.canvas.locked = True

        self.button_play.visible = False
        self.button_pause.visible = True

    def canvas_object_edited(self, canvas, object_name):
        #print("Object <" + str(object_name) + "> was edited")
        x = canvas.elements[object_name].x
        y = canvas.elements[object_name].y
        w = canvas.elements[object_name].w
        h = canvas.elements[object_name].h

        prev_location = self.video_objects[object_name].get_location_at(self.last_video_frame, False)

        keyframe_added = self.video_objects[object_name].set_location_at(self.last_video_frame, self.last_video_time,
                                                                         True, x, y, w, h)

        if keyframe_added:
            # Do not store interpolated locations
            prev_location = None
        else:
            # use a copy of the location provided
            prev_location = VideoObjectLocation.fromLocation(prev_location)

        self.changes_saved = False
        object_location = VideoObjectLocation(True, self.last_video_frame, self.last_video_time, x, y, w, h)

        # check if the object was the last object edited
        if keyframe_added:
            # new key-frame
            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": object_name,
                "old_location": prev_location,
                "new_location": object_location,
            })

        else:
            # edited key-frame, check if same as last change
            if (len(self.undo_stack) > 0 and
                self.undo_stack[-1]["operation"] == "keyframe_edited" and
                self.undo_stack[-1]["object_id"] == object_name and
                self.undo_stack[-1]["new_location"].frame == self.last_video_frame and
                time.time() - self.undo_stack[-1]["time"] < GTContentAnnotator.EditGroupingTime):
                # same object was modified last within n seconds, combine
                self.undo_stack[-1]["new_location"] = object_location
            else:
                # first modification to this object will be added to the top of the stack
                self.undo_stack.append({
                    "operation": "keyframe_edited",
                    "object_id": object_name,
                    "old_location": prev_location,
                    "new_location": object_location,
                    "time": time.time()
                })

        self.update_keyframe_buttons()

    def btn_object_add(self, button):
        # set adding mode
        self.prepare_confirm_input_mode(1, "", None)


    def btn_object_rename(self, button):
        # first, check an object is selected on the list
        selected_name = self.object_selector.selected_option_value
        if selected_name is None:
            # nothing is selected
            return

        # set rename mode
        self.prepare_confirm_input_mode(2, selected_name, None)

    def btn_object_remove(self, button):
        # first, check an object is selected on the list
        selected_name = self.object_selector.selected_option_value
        if selected_name is None:
            # nothing is selected
            return

        # set delete mode
        self.prepare_confirm_input_mode(3, None, "Are you sure?")

    def prepare_confirm_input_mode(self, text_operation, textbox_text, message_text):
        # first, pause the video (if playing)
        self.btn_pause_click(None)

        # Now, change containers
        self.canvas.locked = True
        self.container_object_options.visible = False
        self.container_video_controls.visible = False
        self.container_keyframe_options.visible = False
        self.container_text_input.visible = True

        # text input only visible for adding/renaming video objects
        self.txt_object_name.visible = (text_operation == 1 or text_operation == 2)
        self.label_txt_object_name.visible = (text_operation == 1 or text_operation == 2)

        # Text ...
        if textbox_text is not None:
            self.txt_object_name.updateText(textbox_text)

        # Message Text ...
        if message_text is None:
            self.label_txt_object_error.visible = False
        else:
            self.label_txt_object_error.set_text(message_text)
            self.label_txt_object_error.visible = True

        self.text_operation = text_operation

    def btn_text_operation_accept_click(self, button):
        new_name = self.txt_object_name.text.strip()
        id_name = new_name.lower()

        # name must not be empty
        if new_name == "" and (self.text_operation == 1 or self.text_operation == 2):
            self.label_txt_object_error.set_text("Object name cannot be empty")
            self.label_txt_object_error.visible = True
            return

        if self.text_operation == 1:
            # add, validate ...
            # ... check unique ...
            if id_name in self.video_objects:
                self.label_txt_object_error.set_text("Object name already in use")
                self.label_txt_object_error.visible = True
                return

            # valid... add!
            if self.add_object(id_name, new_name, self.last_video_frame, self.last_video_time, 10, 10, 100, 100):
                location = VideoObjectLocation(True, self.last_video_frame, self.last_video_time, 10, 10, 100, 100)
                self.undo_stack.append({
                    "operation": "object_added",
                    "id": id_name,
                    "name": new_name,
                    "location": location,
                })
            else:
                return

        if self.text_operation == 2:
            # rename
            selected_name = self.object_selector.selected_option_value

            if selected_name != id_name:
                # name changed!, verify ...
                if id_name in self.video_objects:
                    self.label_txt_object_error.set_text("Object name already in use")
                    self.label_txt_object_error.visible = True
                    return

            old_display = self.object_selector.option_display[selected_name]
            if self.rename_object(selected_name, id_name, new_name):
                self.undo_stack.append({
                    "operation": "object_renamed",
                    "old_id": selected_name,
                    "old_display": old_display,
                    "new_id": id_name,
                    "new_display": new_name,
                })
            else:
                return

        if self.text_operation == 3:
            # delete (confirmed)
            selected_name = self.object_selector.selected_option_value

            removed_object = self.video_objects[selected_name]
            removed_display = self.object_selector.option_display[selected_name]
            if self.remove_object(selected_name):
                self.undo_stack.append({
                    "operation": "object_removed",
                    "id": selected_name,
                    "display": removed_display,
                    "object_ref": removed_object,
                })

        if self.text_operation == 4:
            self.return_screen = None
            print("APPLICATION FINISHED / CHANGES LOST")

        self.container_object_options.visible = True
        self.container_video_controls.visible = True
        self.container_keyframe_options.visible = (self.object_selector.selected_option_value is not None)
        self.container_text_input.visible = False
        self.canvas.locked = False

        self.update_keyframe_buttons()

    def btn_text_operation_cancel_click(self, button):
        self.text_operation = 0

        self.container_object_options.visible = True
        self.container_video_controls.visible = True
        self.container_keyframe_options.visible = (self.object_selector.selected_option_value is not None)
        self.container_text_input.visible = False
        self.canvas.locked = False

    def object_selector_option_changed(self, new_value, old_value):
        self.select_object(new_value, 1)

    def canvas_selection_changed(self, object_selected):
        self.select_object(object_selected, 2)

    def select_object(self, new_object, source):
        if source != 1:
            # mark object selector ...
            self.object_selector.change_option_selected(new_object)

        if source != 2:
            # select object in canvas ...
            self.canvas.change_selected_element(new_object)

        self.container_keyframe_options.visible = new_object is not None

        self.update_keyframe_buttons()

    def btn_keyframe_visible_click(self, button):
        self.set_object_keyframe_visible(True)
        self.btn_keyframe_visible.visible = False
        self.btn_keyframe_invisible.visible = True

    def btn_keyframe_invisible_click(self, button):
        self.set_object_keyframe_visible(False)
        self.btn_keyframe_visible.visible = True
        self.btn_keyframe_invisible.visible = False

    def set_object_keyframe_visible(self, is_visible):
        current_frame = self.last_video_frame
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.video_objects[selected_name]
            # next/previous ....
            loc_idx = current_object.find_location_idx(current_frame)

            if not loc_idx >= len(current_object.locations) and current_object.locations[loc_idx].frame == current_frame:
                current_loc = current_object.locations[loc_idx]
                # copy before changing
                old_location = VideoObjectLocation.fromLocation(current_loc)

                #change ...
                current_loc.visible = is_visible
                self.canvas.elements[selected_name].visible = is_visible

                self.changes_saved = False

                # add to undo stack
                self.undo_stack.append({
                    "operation": "keyframe_edited",
                    "object_id": selected_name,
                    "old_location": old_location,
                    "new_location": VideoObjectLocation.fromLocation(current_loc),
                    "time": time.time(),
                })

    def btn_jump_frame_click(self, button):
        self.player.set_player_frame(button.tag, True)

    def btn_keyframe_add_click(self, button):
        current_frame = self.last_video_frame
        current_time = self.last_video_time
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.video_objects[selected_name]

            loc_idx = current_object.find_location_idx(current_frame)

            if loc_idx >= len(current_object.locations):
                # out of boundaries, after last one
                base_loc = current_object.locations[-1]
            else:
                if current_object.locations[0].frame > current_frame:
                    # out of boundaries, before first one
                    base_loc = current_object.locations[0]
                else:
                    # only add if the current frame is not a keyframe already ...
                    if current_object.locations[loc_idx].frame != current_frame:
                        # not key-frame and not out of boundaries
                        base_loc = VideoObjectLocation.interpolate(current_object.locations[loc_idx - 1],
                                                                   current_object.locations[loc_idx], current_frame)
                    else:
                        # already a key-frame
                        return

            current_object.set_location_at(current_frame, current_time, base_loc.visible, base_loc.x, base_loc.y,
                                           base_loc.w, base_loc.h)
            self.canvas.update_element(selected_name, base_loc.x, base_loc.y, base_loc.w, base_loc.h, base_loc.visible)

            self.changes_saved = False
            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": selected_name,
                "old_location": None,
                "new_location": VideoObjectLocation.fromLocation(base_loc),
            })

        self.update_keyframe_buttons()

    def btn_keyframe_del_click(self, button):
        current_frame = self.last_video_frame
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.video_objects[selected_name]
            loc_idx = current_object.find_location_idx(current_frame)

            if (loc_idx < len(current_object.locations) and current_object.locations[loc_idx].frame == current_frame
                and len(current_object.locations) > 1):
                # a key-frame is selected, and is not the only one
                to_delete = current_object.locations[loc_idx]
                del current_object.locations[loc_idx]

                self.changes_saved = False
                self.undo_stack.append({
                    "operation": "keyframe_deleted",
                    "object_id": selected_name,
                    "old_location": VideoObjectLocation.fromLocation(to_delete),
                })

                # update everything ...
                # ... canvas ...
                self.update_canvas_objects()

                # ... key-frame buttons ...
                self.update_keyframe_buttons()


    def update_keyframe_buttons(self):
        current_frame = self.last_video_frame

        selected_name = self.canvas.selected_element

        if selected_name is None:
            # count ...
            self.lbl_keyframe_title.set_text("Object Key-frames: [0]")

            # next/previous ....
            self.lbl_keyframe_prev.set_text("[0]")
            self.lbl_keyframe_next.set_text("[0]")
            # hide everything ...
            self.container_keyframe_options.visible = False
        else:
            current_object = self.video_objects[selected_name]
            # count ...
            self.lbl_keyframe_title.set_text("Object Key-frames: "  + str(len(current_object.locations)))

            # next/previous ....
            loc_idx = current_object.find_location_idx(current_frame)

            # make invisible by default
            self.btn_keyframe_visible.visible = False
            self.btn_keyframe_invisible.visible = False

            self.lbl_keyframe_prev.visible = False
            self.btn_keyframe_prev.visible = False
            self.btn_keyframe_prev.tag = None
            self.lbl_keyframe_next.visible = False
            self.btn_keyframe_next.visible = False
            self.btn_keyframe_next.tag = None

            self.btn_keyframe_add.visible = True
            self.btn_keyframe_del.visible = False

            if loc_idx >= len(current_object.locations):
                # out of boundaries, next is none and prev is last
                self.lbl_keyframe_prev.visible = True
                self.btn_keyframe_prev.visible = True
                self.lbl_keyframe_prev.set_text("[" + str(current_object.locations[-1].frame) + "]")
                self.btn_keyframe_prev.tag = current_object.locations[-1].frame
                self.lbl_keyframe_next.set_text("[X]")

            else:
                if current_object.locations[0].frame > current_frame:
                    # out of boundaries, next is first and prev is None
                    self.lbl_keyframe_prev.set_text("[X]")
                    self.lbl_keyframe_next.visible = True
                    self.btn_keyframe_next.visible = True
                    self.lbl_keyframe_next.set_text("[" + str(current_object.locations[0].frame) + "]")
                    self.btn_keyframe_next.tag = current_object.locations[0].frame
                else:
                    if current_object.locations[loc_idx].frame == current_frame:
                        self.btn_keyframe_add.visible = False
                        self.btn_keyframe_del.visible = len(current_object.locations) > 1

                        # on a key-frame
                        if loc_idx == 0:
                            # no previous ...
                            self.lbl_keyframe_prev.set_text("[X]")
                        else:
                            # previous keyframe is before current frame (which is a keyframe)
                            self.lbl_keyframe_prev.visible = True
                            self.btn_keyframe_prev.visible = True
                            self.lbl_keyframe_prev.set_text("[" + str(current_object.locations[loc_idx - 1].frame) + "]")
                            self.btn_keyframe_prev.tag = current_object.locations[loc_idx - 1].frame

                        if loc_idx == len(current_object.locations) - 1:
                            # no next
                            self.lbl_keyframe_next.set_text("[X]")
                        else:
                            # next keyframe is after current frame (which is a keyframe)
                            self.lbl_keyframe_next.visible = True
                            self.btn_keyframe_next.visible = True
                            self.lbl_keyframe_next.set_text("[" + str(current_object.locations[loc_idx + 1].frame) + "]")
                            self.btn_keyframe_next.tag = current_object.locations[loc_idx + 1].frame

                        # show the corresponding show/hide button ...
                        self.btn_keyframe_invisible.visible = current_object.locations[loc_idx].visible
                        self.btn_keyframe_visible.visible = not current_object.locations[loc_idx].visible
                    else:
                        # not key-frame and not out of boundaries

                        # previous keyframe is closest keyframe
                        self.lbl_keyframe_prev.visible = True
                        self.btn_keyframe_prev.visible = True
                        self.lbl_keyframe_prev.set_text("[" + str(current_object.locations[loc_idx - 1].frame) + "]")
                        self.btn_keyframe_prev.tag = current_object.locations[loc_idx - 1].frame

                        # next keyframe is after current frame (which is a keyframe)
                        self.lbl_keyframe_next.visible = True
                        self.btn_keyframe_next.visible = True
                        self.lbl_keyframe_next.set_text("[" + str(current_object.locations[loc_idx].frame) + "]")
                        self.btn_keyframe_next.tag = current_object.locations[loc_idx].frame

    def update_video_segment_buttons(self):
        self.container_vid_seg_options.visible = self.last_video_frame is not None
        self.container_vid_seg_keyframe_options.visible = self.last_video_frame is not None
        if self.last_video_frame is None:
            return

        # Update the count
        self.lbl_vid_seg_title.set_text("Video Segments: " + str(len(self.video_segments) + 1))

        position = 0
        while position < len(self.video_segments) and self.last_video_frame > self.video_segments[position]:
            position += 1

        if position == len(self.video_segments):
            # at the end ...
            # check if unique segment too..
            if len(self.video_segments) > 0:
                prev_split = self.video_segments[-1]
            else:
                # unique element
                prev_split = 0
            next_split = None

            interval_start = prev_split
            interval_end = self.player.video_player.total_frames
        elif self.video_segments[position] == self.last_video_frame:
            # the exact element
            if position > 0:
                prev_split = self.video_segments[position - 1]
            else:
                prev_split = 0

            interval_start = self.video_segments[position]
            if position + 1 < len(self.video_segments):
                next_split = self.video_segments[position + 1]
                interval_end = next_split
            else:
                next_split = None
                interval_end = self.player.video_player.total_frames
        else:
            # other elements ...
            # check if beginning
            if position > 0:
                prev_split = self.video_segments[position - 1]
            else:
                prev_split = 0
            next_split = self.video_segments[position]

            interval_start = prev_split
            interval_end = next_split

        # prev/next segment
        self.btn_vid_seg_prev.visible = prev_split is not None
        self.lbl_vid_seg_prev.visible = prev_split is not None
        if prev_split is not None:
            self.btn_vid_seg_prev.tag = prev_split
            self.lbl_vid_seg_prev.set_text("[" + str(prev_split) + "]")

        self.btn_vid_seg_next.visible = next_split is not None
        self.lbl_vid_seg_next.visible = next_split is not None
        if next_split is not None:
            self.btn_vid_seg_next.tag = next_split
            self.lbl_vid_seg_next.set_text("[" + str(next_split) + "]")

        # segment split/merge
        not_first_or_last = 0 < self.last_video_frame < self.player.video_player.total_frames - 1
        self.btn_vid_seg_merge.visible = not_first_or_last and self.last_video_frame in self.video_segments
        self.btn_vid_seg_split.visible = not_first_or_last and self.last_video_frame not in self.video_segments

        # Determine the count of key-frames in the current segment
        interval_keyframes = [idx for idx in self.video_segment_keyframes if interval_start <= idx < interval_end]

        position = 0
        while position < len(self.video_segment_keyframes) and self.last_video_frame > self.video_segment_keyframes[position]:
            position += 1

        if position == len(self.video_segment_keyframes):
            # at the end ...
            # check if unique segment too..
            if len(self.video_segment_keyframes) > 0:
                prev_keyframe = self.video_segment_keyframes[-1]
            else:
                # unique element
                prev_keyframe = None
            next_keyframe = None

        elif self.video_segment_keyframes[position] == self.last_video_frame:
            # the exact element
            if position > 0:
                prev_keyframe = self.video_segment_keyframes[position - 1]
            else:
                prev_keyframe = None

            if position + 1 < len(self.video_segment_keyframes):
                next_keyframe = self.video_segment_keyframes[position + 1]
            else:
                next_keyframe = None
        else:
            # other elements ...
            # check if beginning
            if position > 0:
                prev_keyframe = self.video_segment_keyframes[position - 1]
            else:
                prev_keyframe = None
            next_keyframe = self.video_segment_keyframes[position]

        # prev/next segment
        self.btn_vid_seg_keyframe_prev.visible = prev_keyframe is not None
        self.lbl_vid_seg_keyframe_prev.visible = prev_keyframe is not None
        if prev_keyframe is not None:
            self.btn_vid_seg_keyframe_prev.tag = prev_keyframe
            self.lbl_vid_seg_keyframe_prev.set_text("[" + str(prev_keyframe) + "]")

        self.btn_vid_seg_keyframe_next.visible = next_keyframe is not None
        self.lbl_vid_seg_keyframe_next.visible = next_keyframe is not None
        if next_keyframe is not None:
            self.btn_vid_seg_keyframe_next.tag = next_keyframe
            self.lbl_vid_seg_keyframe_next.set_text("[" + str(next_keyframe) + "]")

        # Keyframe add/del
        self.btn_vid_seg_keyframe_del.visible = self.last_video_frame in self.video_segment_keyframes
        self.btn_vid_seg_keyframe_add.visible = self.last_video_frame not in self.video_segment_keyframes

        self.lbl_vid_seg_keyframe_title.set_text("Interval [" + str(interval_start) + ", " + str(interval_end) + ") Key-frames: " +
                                        str(len(interval_keyframes)) + "/" + str(len(self.video_segment_keyframes)))

    def segment_split(self, split_point, add_undo):
        first_or_last = 0 == split_point or split_point >= self.player.video_player.total_frames - 1

        if first_or_last or split_point in self.video_segments:
            # do not split here ...
            return False
        else:
            # add the split point at the current location
            self.video_segments.append(split_point)
            # keep sorted
            self.video_segments = sorted(self.video_segments)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_split",
                    "split_point": split_point,
                })

            return True

    def segment_merge(self, split_point, add_undo):
        if split_point in self.video_segments:
            self.video_segments.remove(split_point)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_merge",
                    "split_point": split_point,
                })

            return True
        else:
            return False

    def btn_vid_seg_split_click(self, button):
        if self.segment_split(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def btn_vid_seg_merge_click(self, button):
        if self.segment_merge(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def segment_keyframe_add(self, frame_index, add_undo):
        if frame_index not in self.video_segment_keyframes:
            # add the key-frame at the current location
            self.video_segment_keyframes.append(frame_index)
            # keep sorted
            self.video_segment_keyframes = sorted(self.video_segment_keyframes)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_keyframe_add",
                    "frame_index": frame_index,
                })

            return True
        else:
            return False

    def segment_keyframe_del(self, frame_index, add_undo):
        if frame_index in self.video_segment_keyframes:
            self.video_segment_keyframes.remove(frame_index)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_keyframe_del",
                    "frame_index": frame_index,
                })

            return True
        else:
            return False

    def btn_vid_seg_keyframe_add_click(self, button):
        if self.segment_keyframe_add(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def btn_vid_seg_keyframe_del_click(self, button):
        if self.segment_keyframe_del(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def generate_metadata_header_xml(self):
        xml_string = "  <Database>" + self.db_name + "</Database>\n"
        xml_string += "  <Lecture>" + self.lecture_title + "</Lecture>\n"
        xml_string += "  <Filename>" + self.output_filename + "</Filename>\n"
        xml_string += "  <VideoFiles>\n"
        for filename in self.video_files:
            xml_string += "  <VideoFile>" + filename + "</VideoFile>\n"
        xml_string += "  </VideoFiles>\n"

        return xml_string

    def generate_export_xml(self, keyframe_times):
        xml_string = "<Annotations>\n"

        # general meta-data
        xml_string += self.generate_metadata_header_xml()

        # segments
        xml_string += self.generate_video_segments_xml()

        # key frames with object data ...
        xml_string += self.generate_keyframes_xml(True, keyframe_times)

        xml_string += "</Annotations>\n"

        return xml_string

    def btn_export_segments_click(self, button):
        # force pause the video???
        self.btn_pause_click(None)

        # check if output directory exists
        main_path = self.output_prefix
        if not os.path.exists(main_path):
            os.mkdir(main_path)

        # check if keyframes sub-directory exists
        keyframes_path = main_path + "/keyframes"
        if not os.path.exists(keyframes_path):
            os.mkdir(keyframes_path)

        # save images for key-frames ....
        current_frame = self.last_video_frame
        frame_times = []
        for keyframe_idx in self.video_segment_keyframes:
            # use the player to extract the video frame ...
            self.player.set_player_frame(keyframe_idx, False)
            frame_img, frame_idx = self.player.video_player.get_frame()

            # keep the key-frame absolute times
            frame_times.append(self.player.video_player.play_abs_position)

            # save image to file ...
            cv2.imwrite(keyframes_path + "/" + str(frame_idx) + ".png", frame_img)

        # restore player current location ...
        self.player.set_player_frame(current_frame, False)

        # save XML string to output file
        xml_data = self.generate_export_xml(frame_times)

        export_filename = main_path + "/segments.xml"
        out_file = open(export_filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("Metadata Saved to: " + export_filename)
