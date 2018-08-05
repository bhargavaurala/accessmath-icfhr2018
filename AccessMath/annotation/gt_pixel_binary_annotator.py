
import cv2
import math
import time
import numpy as np

from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_label import ScreenLabel
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_image import ScreenImage
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll

import scipy.ndimage.measurements as sci_mes
from AccessMath.preprocessing.content.labeler import Labeler

class GTPixelBinaryAnnotator(Screen):
    ModeNavigation = 0
    ModeEdition = 1
    ModeConfirmCancel = 2
    ModeGrowCC_Select = 3
    ModeGrowCC_Growing = 4
    ModeShrinkCC_Select = 5
    ModeShrinkCC_Shrinking = 6

    ViewModeRaw = 0
    ViewModeGray = 1
    ViewModeBinary = 2
    ViewModeSoftCombined = 3
    ViewModeHardCombined = 4

    CCShowBlack = 0
    CCShowColored = 1
    CCShowColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255),
                    (128, 0, 0), (0, 128, 0), (0, 0, 128),
                    (128, 128, 0), (128, 0, 128), (0, 128, 128),
                    (255, 128, 0), (255, 0, 128), (0, 255, 128),
                    (128, 255, 0), (128, 0, 255), (0, 128, 255),]

    CCExpansionDistance = 1

    def __init__(self, size, raw_input, binary_input, parent_screen=None):
        Screen.__init__(self, "Ground Truth Binary Pixel Annotation Interface", size)

        self.small_mode = self.height < 800

        # base images
        self.base_raw = raw_input.copy()
        tempo_gray = cv2.cvtColor(raw_input, cv2.COLOR_RGB2GRAY)
        self.base_gray = np.zeros((self.base_raw.shape[0], self.base_raw.shape[1], 3), dtype=np.uint8)
        self.base_gray[:, :, 0] = tempo_gray.copy()
        self.base_gray[:, :, 1] = tempo_gray.copy()
        self.base_gray[:, :, 2] = tempo_gray.copy()

        self.base_binary = binary_input.copy()

        self.base_ccs = None
        self.base_cc_image = None    # colored version

        # stats
        self.stats_pixels_white = 0
        self.stats_pixels_black = 0
        self.stats_pixels_ratio = 0

        self.stats_cc_count = 0
        self.stats_cc_size_min = 0
        self.stats_cc_size_max = 0
        self.stats_cc_size_mean = 0
        self.stats_cc_size_median = 0

        # for automatic CC expansion
        self.selected_CC = None
        self.CC_expansion_pixels = None
        self.count_expand = 0
        # ... shrinking ...
        self.CC_shrinking_pixels = None
        self.count_shrink = 0

        # view params
        self.view_scale = 1.0
        self.max_scale = None
        self.view_mode = GTPixelBinaryAnnotator.ViewModeBinary
        self.editor_mode = GTPixelBinaryAnnotator.ModeNavigation
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowColored
        self.min_highlight_size = 0

        # appearance parameters
        general_background = (125, 40, 20)
        text_color = (255, 255, 255)
        button_text_color = (50, 35, 20)
        button_back_color = (228, 228, 228)
        self.elements.back_color = general_background

        # add elements....
        # right panel button size and horizontal locations
        container_width = 330

        button_width = 190
        button_left = (container_width - button_width) / 2

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        button_3_width = 100
        button_3_left = 10
        button_3_middle = (container_width - button_3_width) / 2
        button_3_right = container_width - button_3_width - 10

        button_4_width = 75
        button_4_left_1 = int(container_width * 0.125) - button_4_width / 2
        button_4_left_2 = int(container_width * 0.375) - button_4_width / 2
        button_4_left_3 = int(container_width * 0.625) - button_4_width / 2
        button_4_left_4 = int(container_width * 0.875) - button_4_width / 2

        # View panel with Zoom control buttons
        self.container_zoom_buttons = ScreenContainer("container_zoom_buttons", (container_width, 80),
                                                      back_color=general_background)
        self.container_zoom_buttons.position = (self.width - container_width - 10, 10)
        self.elements.append(self.container_zoom_buttons)

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", 21, container_width - 10, 1)
        self.lbl_zoom.position = (5, 5)
        self.lbl_zoom.set_background(general_background)
        self.lbl_zoom.set_color(text_color)
        self.container_zoom_buttons.append(self.lbl_zoom)

        self.btn_zoom_reduce = ScreenButton("btn_zoom_reduce", "[ - ]", 21, 90)
        self.btn_zoom_reduce.set_colors(button_text_color, button_back_color)
        self.btn_zoom_reduce.position = (10, self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_reduce.click_callback = self.btn_zoom_reduce_click
        self.container_zoom_buttons.append(self.btn_zoom_reduce)

        self.btn_zoom_increase = ScreenButton("btn_zoom_increase", "[ + ]", 21, 90)
        self.btn_zoom_increase.set_colors(button_text_color, button_back_color)
        self.btn_zoom_increase.position = (self.container_zoom_buttons.width - self.btn_zoom_increase.width - 10,
                                           self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_increase.click_callback = self.btn_zoom_increase_click
        self.container_zoom_buttons.append(self.btn_zoom_increase)

        self.btn_zoom_zero = ScreenButton("btn_zoom_zero", "100%", 21, 90)
        self.btn_zoom_zero.set_colors(button_text_color, button_back_color)
        self.btn_zoom_zero.position = ((self.container_zoom_buttons.width - self.btn_zoom_zero.width) / 2,
                                       self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_zero.click_callback = self.btn_zoom_zero_click
        self.container_zoom_buttons.append(self.btn_zoom_zero)

        # ===========================
        self.container_cc_expansion = ScreenContainer("container_cc_expansion", (container_width, 150), back_color=general_background)
        self.container_cc_expansion.position = (self.container_zoom_buttons.get_left(), self.container_zoom_buttons.get_bottom() + 10)
        self.elements.append(self.container_cc_expansion)

        self.lbl_cc_expansion = ScreenLabel("lbl_cc_expansion", "Expand CC, Pixels = " + str(self.count_expand), 21, container_width - 10, 1)
        self.lbl_cc_expansion.position = (5, 5)
        self.lbl_cc_expansion.set_background(general_background)
        self.lbl_cc_expansion.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_cc_expansion)

        self.expansion_scroll = ScreenHorizontalScroll("expansion_scroll", 0, 100, 0, 10)
        self.expansion_scroll.position = (5, self.lbl_cc_expansion.get_bottom() + 10)
        self.expansion_scroll.width = container_width - 10
        self.expansion_scroll.scroll_callback = self.expansion_scroll_change
        self.container_cc_expansion.append(self.expansion_scroll)

        self.lbl_expand_confirm = ScreenLabel("lbl_confirm", "Expand CC Pixels", 21, container_width - 10, 1)
        self.lbl_expand_confirm.position = (5, self.expansion_scroll.get_bottom() + 15)
        self.lbl_expand_confirm.set_background(general_background)
        self.lbl_expand_confirm.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_expand_confirm)

        self.btn_expand_accept = ScreenButton("btn_expand_accept", "Accept", 21, 130)
        self.btn_expand_accept.set_colors(button_text_color, button_back_color)
        self.btn_expand_accept.position = (10, self.lbl_expand_confirm.get_bottom() + 10)
        self.btn_expand_accept.click_callback = self.btn_expand_accept_click
        self.container_cc_expansion.append(self.btn_expand_accept)

        self.btn_expand_cancel = ScreenButton("btn_expand_cancel", "Cancel", 21, 130)
        self.btn_expand_cancel.set_colors(button_text_color, button_back_color)
        self.btn_expand_cancel.position = (container_width - self.btn_expand_cancel.width - 10, self.lbl_expand_confirm.get_bottom() + 10)
        self.btn_expand_cancel.click_callback = self.btn_expand_cancel_click
        self.container_cc_expansion.append(self.btn_expand_cancel)
        self.container_cc_expansion.visible = False

        # =======================================

        # View panel with view control buttons
        self.container_view_buttons = ScreenContainer("container_view_buttons", (container_width, 200),
                                                      back_color=general_background)
        self.container_view_buttons.position = (self.width - container_width - 10, self.container_zoom_buttons.get_bottom() + 5)
        self.elements.append(self.container_view_buttons)

        # ===========================
        self.lbl_views = ScreenLabel("lbl_zoom", "Views", 21, button_3_width, 1)
        self.lbl_views.position = (button_3_left, 5)
        self.lbl_views.set_background(general_background)
        self.lbl_views.set_color(text_color)
        self.container_view_buttons.append(self.lbl_views)

        self.btn_view_raw = ScreenButton("btn_view_raw", "Raw", 21, button_3_width)
        self.btn_view_raw.set_colors(button_text_color, button_back_color)
        self.btn_view_raw.position = (button_3_middle, 5)
        self.btn_view_raw.click_callback = self.btn_view_raw_click
        self.container_view_buttons.append(self.btn_view_raw)

        self.btn_view_gray = ScreenButton("btn_view_gray", "Gray", 21, button_3_width)
        self.btn_view_gray.set_colors(button_text_color, button_back_color)
        self.btn_view_gray.position = (button_3_right, 5)
        self.btn_view_gray.click_callback = self.btn_view_gray_click
        self.container_view_buttons.append(self.btn_view_gray)

        self.btn_view_binary = ScreenButton("btn_view_binary", "Binary", 21, button_3_width)
        self.btn_view_binary.set_colors(button_text_color, button_back_color)
        self.btn_view_binary.position = (button_3_left, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_binary.click_callback = self.btn_view_bin_click
        self.container_view_buttons.append(self.btn_view_binary)

        self.btn_view_combo_hard = ScreenButton("btn_view_combo_hard", "CC Hard", 21, button_3_width)
        self.btn_view_combo_hard.set_colors(button_text_color, button_back_color)
        self.btn_view_combo_hard.position = (button_3_middle, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_combo_hard.click_callback = self.btn_view_combo_hard_click
        self.container_view_buttons.append(self.btn_view_combo_hard)

        self.btn_view_combo_soft = ScreenButton("btn_view_combo_soft", "CC Soft", 21, button_3_width)
        self.btn_view_combo_soft.set_colors(button_text_color, button_back_color)
        self.btn_view_combo_soft.position = (button_3_right, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_combo_soft.click_callback = self.btn_view_combo_soft_click
        self.container_view_buttons.append(self.btn_view_combo_soft)

        # ===========================
        self.lbl_show_cc = ScreenLabel("lbl_show_cc", "Display CC", 21, button_3_width, 1)
        self.lbl_show_cc.position = (button_3_left, self.btn_view_combo_soft.get_bottom() + 20)
        self.lbl_show_cc.set_background(general_background)
        self.lbl_show_cc.set_color(text_color)
        self.container_view_buttons.append(self.lbl_show_cc)

        self.btn_show_cc_black = ScreenButton("btn_show_cc_black", "Black", 21, button_3_width)
        self.btn_show_cc_black.set_colors(button_text_color, button_back_color)
        self.btn_show_cc_black.position = (button_3_middle, self.btn_view_combo_soft.get_bottom() + 10)
        self.btn_show_cc_black.click_callback = self.btn_show_cc_black_click
        self.container_view_buttons.append(self.btn_show_cc_black)

        self.btn_show_cc_colored = ScreenButton("btn_show_cc_colored", "Colored", 21, button_3_width)
        self.btn_show_cc_colored.set_colors(button_text_color, button_back_color)
        self.btn_show_cc_colored.position = (button_3_right, self.btn_view_combo_soft.get_bottom() + 10)
        self.btn_show_cc_colored.click_callback = self.btn_show_cc_colored_click
        self.container_view_buttons.append(self.btn_show_cc_colored)

        # ===========================
        self.lbl_small_highlight = ScreenLabel("lbl_small_highlight", "Highlight CCs smaller than: " + str(self.min_highlight_size), 21, 290, 1)
        self.lbl_small_highlight.position = (5, self.btn_show_cc_black.get_bottom() + 20)
        self.lbl_small_highlight.set_background(general_background)
        self.lbl_small_highlight.set_color(text_color)
        self.container_view_buttons.append(self.lbl_small_highlight)

        self.highlight_scroll = ScreenHorizontalScroll("highlight_scroll", 0, 100, 0, 10)
        self.highlight_scroll.position = (5, self.lbl_small_highlight.get_bottom() + 10)
        self.highlight_scroll.width = container_width - 10
        self.highlight_scroll.scroll_callback = self.highlight_scroll_change
        self.container_view_buttons.append(self.highlight_scroll)

        # ===========================
        self.container_edition_mode = ScreenContainer("container_edition_mode", (container_width, 60),
                                                      back_color=general_background)
        self.container_edition_mode.position = (self.width - container_width - 10, self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_edition_mode)

        self.btn_edition_start = ScreenButton("btn_edition_start", "Edit Pixels", 21, button_3_width)
        self.btn_edition_start.set_colors(button_text_color, button_back_color)
        self.btn_edition_start.position = (button_3_left, 5)
        self.btn_edition_start.click_callback = self.btn_edition_start_click
        self.container_edition_mode.append(self.btn_edition_start)

        self.btn_edition_stop = ScreenButton("btn_edition_stop", "Stop Editing Pixels", 21, button_width)
        self.btn_edition_stop.set_colors(button_text_color, button_back_color)
        self.btn_edition_stop.position = (button_left, 5)
        self.btn_edition_stop.click_callback = self.btn_edition_stop_click
        self.btn_edition_stop.visible = False
        self.container_edition_mode.append(self.btn_edition_stop)

        self.btn_edition_expand = ScreenButton("btn_edition_expand", "Expand CC", 21, button_3_width)
        self.btn_edition_expand.set_colors(button_text_color, button_back_color)
        self.btn_edition_expand.position = (button_3_middle, 5)
        self.btn_edition_expand.click_callback = self.btn_edition_expand_click
        self.container_edition_mode.append(self.btn_edition_expand)

        self.btn_edition_shrink = ScreenButton("btn_edition_shrink", "Shrink\nCC", 21, button_3_width)
        self.btn_edition_shrink.set_colors(button_text_color, button_back_color)
        self.btn_edition_shrink.position = (button_3_right, 5)
        self.btn_edition_shrink.click_callback = self.btn_edition_shrink_click
        self.container_edition_mode.append(self.btn_edition_shrink)

        # ===========================
        # Panel with confirmation buttons (Message, Accept, Cancel)
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (300, 70), back_color=general_background)
        self.container_confirm_buttons.position = (self.container_view_buttons.get_left(), self.container_edition_mode.get_bottom() + 10)
        self.elements.append(self.container_confirm_buttons)

        self.lbl_confirm = ScreenLabel("lbl_confirm", "Exit without saving?", 21, 290, 1)
        self.lbl_confirm.position = (5, 5)
        self.lbl_confirm.set_background(general_background)
        self.lbl_confirm.set_color(text_color)
        self.container_confirm_buttons.append(self.lbl_confirm)

        self.btn_confirm_accept = ScreenButton("btn_confirm_accept", "Accept", 21, 130)
        self.btn_confirm_accept.set_colors(button_text_color, button_back_color)
        self.btn_confirm_accept.position = (10, self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_accept.click_callback = self.btn_confirm_accept_click
        self.container_confirm_buttons.append(self.btn_confirm_accept)

        self.btn_confirm_cancel = ScreenButton("btn_confirm_cancel", "Cancel", 21, 130)
        self.btn_confirm_cancel.set_colors(button_text_color, button_back_color)
        self.btn_confirm_cancel.position = (self.container_confirm_buttons.width - self.btn_confirm_cancel.width - 10,
                                            self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_cancel.click_callback = self.btn_confirm_cancel_click
        self.container_confirm_buttons.append(self.btn_confirm_cancel)
        self.container_confirm_buttons.visible = False

        # =============================
        stats_background = (60, 20, 10)
        self.container_stats = ScreenContainer("container_stats", (container_width, 160), back_color=stats_background)
        self.container_stats.position = (self.width - container_width - 10, self.container_edition_mode.get_bottom() + 5)
        self.elements.append(self.container_stats)

        self.lbl_pixel_stats = ScreenLabel("lbl_pixel_stats", "Pixel Stats", 21, container_width - 10, 1)
        self.lbl_pixel_stats.position = (5, 5)
        self.lbl_pixel_stats.set_background(stats_background)
        self.lbl_pixel_stats.set_color(text_color)
        self.container_stats.append(self.lbl_pixel_stats)

        self.lbl_pixels_white = ScreenLabel("lbl_pixels_white", "White:\n1000000", 21, button_3_width, 1)
        self.lbl_pixels_white.position = (button_3_left, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_white.set_background(stats_background)
        self.lbl_pixels_white.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_white)

        self.lbl_pixels_black = ScreenLabel("lbl_pixels_black", "Black:\n1000000", 21, button_3_width, 1)
        self.lbl_pixels_black.position = (button_3_middle, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_black.set_background(stats_background)
        self.lbl_pixels_black.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_black)

        self.lbl_pixels_ratio = ScreenLabel("lbl_pixels_ratio", "Ratio:\n0.00000", 21, button_3_width, 1)
        self.lbl_pixels_ratio.position = (button_3_right, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_ratio.set_background(stats_background)
        self.lbl_pixels_ratio.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_ratio)

        self.lbl_cc_stats = ScreenLabel("lbl_cc_stats", "CC Stats", 21, container_width - 10, 1)
        self.lbl_cc_stats.position = (5, self.lbl_pixels_ratio.get_bottom() + 10)
        self.lbl_cc_stats.set_background(stats_background)
        self.lbl_cc_stats.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats)

        self.lbl_cc_stats_count = ScreenLabel("lbl_cc_stats_count", "Total CC: 0", 21, container_width - 10, 1)
        self.lbl_cc_stats_count.position = (5, self.lbl_cc_stats.get_bottom() + 10)
        self.lbl_cc_stats_count.set_background(stats_background)
        self.lbl_cc_stats_count.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_count)

        self.lbl_cc_stats_min = ScreenLabel("lbl_cc_stats_min", "Min:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_min.position = (button_4_left_1, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_min.set_background(stats_background)
        self.lbl_cc_stats_min.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_min)

        self.lbl_cc_stats_max = ScreenLabel("lbl_cc_stats_max", "Max:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_max.position = (button_4_left_2, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_max.set_background(stats_background)
        self.lbl_cc_stats_max.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_max)

        self.lbl_cc_stats_mean = ScreenLabel("lbl_cc_stats_mean", "Mean:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_mean.position = (button_4_left_3, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_mean.set_background(stats_background)
        self.lbl_cc_stats_mean.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_mean)

        self.lbl_cc_stats_median = ScreenLabel("lbl_cc_stats_median", "Median:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_median.position = (button_4_left_4, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_median.set_background(stats_background)
        self.lbl_cc_stats_median.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_median)

        # ============================================================================
        # Panel with state buttons (Undo, Redo, return accept, return cancel)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 120), general_background)
        self.container_state_buttons.position = (self.container_stats.get_left(), self.container_stats.get_bottom() + 10)
        self.elements.append(self.container_state_buttons)

        self.btn_undo = ScreenButton("btn_undo", "Undo", 21, button_2_width)
        self.btn_undo.set_colors(button_text_color, button_back_color)
        self.btn_undo.position = (button_2_left, 5)
        self.btn_undo.click_callback = self.btn_undo_click
        self.container_state_buttons.append(self.btn_undo)

        self.btn_redo = ScreenButton("btn_redo", "Redo", 21, button_2_width)
        self.btn_redo.set_colors(button_text_color, button_back_color)
        self.btn_redo.position = (button_2_right, 5)
        self.btn_redo.click_callback = self.btn_redo_click
        self.container_state_buttons.append(self.btn_redo)

        # Secondary screen mode
        # Add Cancel Button
        self.btn_return_cancel = ScreenButton("btn_return_cancel", "Cancel", 21, button_2_width)
        self.btn_return_cancel.set_colors(button_text_color, button_back_color)
        self.btn_return_cancel.position = (button_2_left, self.btn_redo.get_bottom() + 30)
        self.btn_return_cancel.click_callback = self.btn_return_cancel_click
        self.container_state_buttons.append(self.btn_return_cancel)

        # Add Accept Button
        self.btn_return_accept = ScreenButton("btn_return_accept", "Accept", 21, button_2_width)
        self.btn_return_accept.set_colors(button_text_color, button_back_color)
        self.btn_return_accept.position = (button_2_right, self.btn_redo.get_bottom() + 30)
        self.btn_return_accept.click_callback = self.btn_return_accept_click
        self.container_state_buttons.append(self.btn_return_accept)

        # ============================================================================
        image_width = self.width - self.container_view_buttons.width - 30
        image_height = self.height - 20
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, 10)
        self.elements.append(self.container_images)

        # ... image objects ...
        tempo_blank = np.zeros((50, 50, 3), np.uint8)
        tempo_blank[:, :, :] = 255
        self.img_main = ScreenImage("img_raw", tempo_blank, 0, 0, True, cv2.INTER_NEAREST)
        self.img_main.position = (0, 0)
        self.img_main.mouse_button_down_callback = self.img_mouse_down
        self.img_main.mouse_motion_callback = self.img_mouse_motion
        self.container_images.append(self.img_main)

        self.pre_edition_binary = None
        self.undo_stack = []
        self.redo_stack = []

        self.finished_callback = None
        self.parent_screen = parent_screen

        self.last_motion_set_x = None
        self.last_motion_set_y = None
        self.last_motion_polarity = None

        self.update_cc_info()
        self.update_current_view(True)


    def update_current_view(self, resized=False, region=None):
        if region is None:
            if self.view_mode == GTPixelBinaryAnnotator.ViewModeGray:
                base_image = self.base_gray
            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeBinary:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

                    base_image[:, :, 0] = self.base_binary
                    base_image[:, :, 1] = self.base_binary
                    base_image[:, :, 2] = self.base_binary
                else:
                    base_image = self.base_cc_image.copy()
                    base_image[self.base_binary > 0, :] = 255

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeHardCombined:
                base_image = self.base_gray.copy()

                # create hard combined
                inverse_binary_mask = self.base_binary == 0
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image[inverse_binary_mask, :] = 0
                else:
                    base_image[inverse_binary_mask, :] = self.base_cc_image[inverse_binary_mask, :].copy()

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeSoftCombined:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

                    scaled_black = 1 - (self.base_binary / 255)
                    base_image[:, :, 0] = scaled_black
                else:
                    base_image = self.base_cc_image.copy()


                # replace the empty channels on each CC with original CC grayscale
                for channel in range(3):
                    inverse_binary_mask = base_image[:, :, channel] == 0

                    base_image[inverse_binary_mask, channel] = self.base_gray[inverse_binary_mask, 0].copy()
            else:
                base_image = self.base_raw.copy()
        else:
            start_x = max(region[0], 0)
            end_x = min(region[1] + 1, self.base_raw.shape[1])
            start_y = max(region[2], 0)
            end_y = min(region[3] + 1, self.base_raw.shape[0])

            # create empty image
            base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

            if self.view_mode == GTPixelBinaryAnnotator.ViewModeGray:
                base_image[start_y:end_y, start_x:end_x] = self.base_gray[start_y:end_y, start_x:end_x].copy()

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeBinary:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:

                    base_image[start_y:end_y, start_x:end_x, 0] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                    base_image[start_y:end_y, start_x:end_x, 1] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                    base_image[start_y:end_y, start_x:end_x, 2] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                else:
                    base_image[start_y:end_y, start_x:end_x] = self.base_cc_image[start_y:end_y, start_x:end_x].copy()

                    cut_mask = self.base_binary[start_y:end_y, start_x:end_x] > 0
                    tempo_cut = base_image[start_y:end_y, start_x:end_x]
                    tempo_cut[cut_mask, :] = 255

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeHardCombined:
                base_image[start_y:end_y, start_x:end_x] = self.base_gray[start_y:end_y, start_x:end_x].copy()

                # create hard combined
                inverse_binary_mask = self.base_binary[start_y:end_y, start_x:end_x] == 0
                tempo_cut = base_image[start_y:end_y, start_x:end_x]

                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    tempo_cut[inverse_binary_mask, :] = 0
                else:
                    cc_image_cut = self.base_cc_image[start_y:end_y, start_x:end_x].copy()
                    tempo_cut[inverse_binary_mask, :] = cc_image_cut[inverse_binary_mask, :]

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeSoftCombined:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:

                    scaled_black = 1 - (self.base_binary[start_y:end_y, start_x:end_x] / 255)
                    base_image[start_y:end_y, start_x:end_x, 0] = scaled_black
                else:
                    base_image[start_y:end_y, start_x:end_x] = self.base_cc_image[start_y:end_y, start_x:end_x].copy()

                # replace the empty channels on each CC with original CC grayscale
                base_cut = base_image[start_y:end_y, start_x:end_x, :]
                gray_cut = self.base_gray[start_y:end_y, start_x:end_x, 0].copy()
                for channel in range(3):
                    inverse_binary_mask = base_image[start_y:end_y, start_x:end_x, channel] == 0

                    base_cut[inverse_binary_mask, channel] = gray_cut[inverse_binary_mask].copy()
                    #base_image[inverse_binary_mask, channel] = self.base_gray[inverse_binary_mask, 0].copy()
            else:
                base_image[start_y:end_y, start_x:end_x] = self.base_raw[start_y:end_y, start_x:end_x].copy()

        h, w, c = base_image.shape

        modified_image = base_image.copy()

        if self.selected_CC is not None:
            # mark selected CC ...
            cc = self.selected_CC
            current_cut = modified_image[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_mask = cc.img > 0
            current_cut[cc_mask, 0] = 255

            # Highlight pixels for current threshold
            if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
                for value, y, x in self.CC_expansion_pixels[:self.count_expand]:
                    modified_image[y, x, 1] = 1
            elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
                for value, y, x in self.CC_shrinking_pixels[:self.count_shrink]:
                    modified_image[y, x, 1] = 1

        # show highlighted small CC (if any)
        if self.min_highlight_size > 0:
            for cc in self.base_ccs:
                if cc.size < self.min_highlight_size:
                    # print(str((cc.getCenter(), cc.size, cc.min_x, cc.max_x, cc.min_y, cc.max_y)))
                    # compute highlight base radius
                    base_radius = math.sqrt(math.pow(cc.getWidth() / 2, 2.0) + math.pow(cc.getHeight() / 2, 2.0))
                    highlight_radius = int(base_radius * 3)

                    cc_cx, cc_cy = cc.getCenter()
                    cv2.circle(modified_image, (int(cc_cx), int(cc_cy)), highlight_radius, (255, 0, 0), 2)

        if region is None:
            # resize ...
            new_res = (int(w * self.view_scale), int(h * self.view_scale))
            modified_image = cv2.resize(modified_image, new_res, interpolation=cv2.INTER_NEAREST)

            # add grid
            if self.view_scale >= 4.0:
                int_scale = int(self.view_scale)
                density = 2
                modified_image[int_scale - 1::int_scale, ::density, :] = 128
                modified_image[::density, int_scale - 1::int_scale, :] = 128

            # replace/update image
            self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        else:
            new_res = (int((end_x - start_x) * self.view_scale), int((end_y - start_y) * self.view_scale))
            portion_cut = cv2.resize(modified_image[start_y:end_y, start_x:end_x, :], new_res, interpolation=cv2.INTER_NEAREST)

            # add grid
            if self.view_scale >= 4.0:
                int_scale = int(self.view_scale)
                density = 2

                old_off = int_scale - 1
                new_off_x = math.ceil((start_x * self.view_scale - old_off) / int_scale) * int_scale + old_off - start_x * self.view_scale
                new_off_y = math.ceil((start_y * self.view_scale - old_off) / int_scale) * int_scale + old_off - start_y * self.view_scale
                portion_cut[new_off_y::int_scale, ::density, :] = 128
                portion_cut[::density, new_off_x::int_scale, :] = 128

            # update region ....
            self.img_main.update_image_region(portion_cut, (int(start_x * self.view_scale), int(start_y * self.view_scale)))

        if resized:
            self.container_images.recalculate_size()

    def update_cc_info(self):
        h, w, _ = self.base_raw.shape

        fake_age = np.zeros((h, w), dtype=np.float32)
        self.base_ccs = Labeler.extractSpatioTemporalContent((255 - self.base_binary), fake_age, False)

        self.base_cc_image = np.zeros((h, w, 3), dtype=np.uint8)

        tempo_sizes = []
        for idx, cc in enumerate(self.base_ccs):
            n_colors = len(GTPixelBinaryAnnotator.CCShowColors)
            cc_color = GTPixelBinaryAnnotator.CCShowColors[idx % n_colors]

            current_cut = self.base_cc_image[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_mask = cc.img > 0

            current_cut[cc_mask, 0] += cc_color[0]
            current_cut[cc_mask, 1] += cc_color[1]
            current_cut[cc_mask, 2] += cc_color[2]

            tempo_sizes.append(cc.size)

        # self.base_cc_mask = np.sum(self.base_cc_image, 2) == 0
        total_pixels = h * w
        self.stats_pixels_white = self.base_binary.sum() / 255
        self.stats_pixels_black = total_pixels - self.stats_pixels_white
        self.stats_pixels_ratio = self.stats_pixels_black / total_pixels

        cc_sizes = np.array(tempo_sizes)
        self.stats_cc_count = len(self.base_ccs)
        self.stats_cc_size_min = cc_sizes.min()
        self.stats_cc_size_max = cc_sizes.max()
        self.stats_cc_size_mean = cc_sizes.mean()
        self.stats_cc_size_median = np.median(cc_sizes)

        # update interface ...
        self.lbl_pixels_white.set_text("White:\n" + str(self.stats_pixels_white))
        self.lbl_pixels_black.set_text("Black:\n" + str(self.stats_pixels_black))
        self.lbl_pixels_ratio.set_text("Ratio:\n{0:.5f}".format(self.stats_pixels_ratio))

        self.lbl_cc_stats_count.set_text("Total CC: {0}".format(self.stats_cc_count))
        self.lbl_cc_stats_min.set_text("Min:\n{0}".format(self.stats_cc_size_min))
        self.lbl_cc_stats_max.set_text("Max:\n{0}".format(self.stats_cc_size_max))
        self.lbl_cc_stats_mean.set_text("Mean:\n{0:.2f}".format(self.stats_cc_size_mean))
        self.lbl_cc_stats_median.set_text("Median:\n{0:.2f}".format(self.stats_cc_size_median))

    def update_view_scale(self, new_scale):
        prev_scale = self.view_scale

        if (new_scale < 0.25) or (self.max_scale is not None and new_scale > self.max_scale):
            # below minimum or above maximum
            return

        self.view_scale = new_scale

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
        try:
            self.update_current_view(True)
        except:
            # rescale failed ...
            # restore previous scale
            self.view_scale = prev_scale
            self.max_scale = prev_scale
            self.update_current_view(True)
            print("Maximum zoom is {0:.2f}%".format(self.max_scale * 100))


        # set offsets
        if self.container_images.v_scroll.active and 0 <= new_off_y <= self.container_images.v_scroll.max:
            self.container_images.v_scroll.value = new_off_y
        if self.container_images.h_scroll.active and 0 <= new_off_x <= self.container_images.h_scroll.max:
            self.container_images.h_scroll.value = new_off_x

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_zoom_reduce_click(self, button):
        if self.view_scale <= 1.0:
            self.update_view_scale(self.view_scale - 0.25)
        else:
            self.update_view_scale(self.view_scale - 1.0)

    def btn_zoom_increase_click(self, button):
        if self.view_scale < 1.0:
            self.update_view_scale(self.view_scale + 0.25)
        else:
            self.update_view_scale(self.view_scale + 1.0)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def btn_view_raw_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeRaw
        self.update_current_view()

    def btn_view_gray_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeGray
        self.update_current_view()

    def btn_view_bin_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeBinary
        self.update_current_view()

    def btn_view_combo_soft_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
        self.update_current_view()

    def btn_view_combo_hard_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeHardCombined
        self.update_current_view()

    def btn_show_cc_black_click(self, button):
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
        self.update_current_view()

    def btn_show_cc_colored_click(self, button):
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowColored
        self.update_current_view()

    def highlight_scroll_change(self, scroll):
        self.min_highlight_size = int(scroll.value)
        self.lbl_small_highlight.set_text("Highlight CCs smaller than: " + str(self.min_highlight_size))

        self.update_current_view(False)

    def set_editor_mode(self, new_mode):
        self.editor_mode = new_mode

        self.container_confirm_buttons.visible = (new_mode == GTPixelBinaryAnnotator.ModeConfirmCancel)

        self.container_state_buttons.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)
        self.container_stats.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)
        self.btn_edition_start.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)
        self.btn_edition_stop.visible = (new_mode == GTPixelBinaryAnnotator.ModeEdition)

        self.container_edition_mode.visible = (new_mode != GTPixelBinaryAnnotator.ModeGrowCC_Select and
                                               new_mode != GTPixelBinaryAnnotator.ModeGrowCC_Growing and
                                               new_mode != GTPixelBinaryAnnotator.ModeShrinkCC_Select and
                                               new_mode != GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

        self.container_view_buttons.visible = (new_mode != GTPixelBinaryAnnotator.ModeGrowCC_Growing and
                                               new_mode != GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

        self.container_cc_expansion.visible = (new_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing or
                                               new_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

        self.btn_edition_expand.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)
        self.btn_edition_shrink.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)

    def btn_edition_start_click(self, button):
        self.pre_edition_binary = self.base_binary.copy()

        self.set_editor_mode(GTPixelBinaryAnnotator.ModeEdition)

    def btn_edition_stop_click(self, button):
        self.undo_stack.append({
            "operation": "pixels_edited",
            "prev_state": self.pre_edition_binary.copy(),
            "new_state": self.base_binary.copy(),
        })

        self.update_cc_info()
        self.update_current_view(False)
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False

        if to_undo["operation"] == "pixels_edited":
            # restore previous state
            self.base_binary = to_undo["prev_state"].copy()
            success = True

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update interface ...
            self.update_cc_info()
            self.update_current_view(False)
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

        # copy last operation
        to_redo = self.redo_stack[-1]

        success = False

        if to_redo["operation"] == "pixels_edited":
            # restore new state
            self.base_binary = to_redo["new_state"].copy()
            success = True

        # removing ...
        if success:
            self.undo_stack.append(to_redo)
            del self.redo_stack[-1]

            # update interface ...
            self.update_cc_info()
            self.update_current_view(False)
        else:
            print("Action could not be re-done")

    def btn_return_accept_click(self, button):
        if self.finished_callback is not None:
            # accept ...
            self.finished_callback(True, self.base_binary)

        self.return_screen = self.parent_screen

    def btn_return_cancel_click(self, button):
        if len(self.undo_stack) == 0:
            if self.finished_callback is not None:
                # accept ...
                self.finished_callback(False, None)

            self.return_screen = self.parent_screen
        else:
            self.set_editor_mode(GTPixelBinaryAnnotator.ModeConfirmCancel)


    def img_mouse_down(self, img_object, pos, button):
        # ... first, get click location on original image space
        scaled_x, scaled_y = pos
        click_x = int(scaled_x / self.view_scale)
        click_y = int(scaled_y / self.view_scale)

        if click_x < 0 or click_y < 0 or click_x >= self.base_raw.shape[1] or click_y >= self.base_raw.shape[0]:
            # out of boundaries
            return

        if button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:
            # invert pixel ...
            self.base_binary[click_y, click_x] = 255 - self.base_binary[click_y, click_x]

            if self.base_binary[click_y, click_x] > 0:
                self.base_cc_image[click_y, click_x, :] = 0
            else:
                self.base_cc_image[click_y, click_x, 0] = 1
                self.base_cc_image[click_y, click_x, 1] = 0
                self.base_cc_image[click_y, click_x, 2] = 0

            self.last_motion_set_x = click_x
            self.last_motion_set_y = click_y
            self.last_motion_polarity = self.base_binary[click_y, click_x]

            self.update_current_view(False, (click_x - 1, click_x + 1, click_y - 1, click_y + 1))

        elif button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Select:
            for cc in self.base_ccs:
                if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                    rel_offset_x = click_x - cc.min_x
                    rel_offset_y = click_y - cc.min_y

                    if cc.img[rel_offset_y, rel_offset_x] > 0:
                        # cv2.imshow("selected", cc.img)
                        # select CC and move to next mode
                        self.pre_edition_binary = self.base_binary.copy()
                        self.selected_CC = cc

                        # compute CC stats ...
                        cc_cut = self.base_gray[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1, 0].copy()
                        cc_mask = cc.img > 0
                        avg_cc_luminosity = cc_cut[cc_mask].mean()

                        # compute expansion
                        exp_min_x = min(GTPixelBinaryAnnotator.CCExpansionDistance, cc.min_x)
                        exp_max_x = min(GTPixelBinaryAnnotator.CCExpansionDistance, self.base_raw.shape[1] - 1 - cc.max_x)
                        exp_min_y = min(GTPixelBinaryAnnotator.CCExpansionDistance, cc.min_y)
                        exp_max_y = min(GTPixelBinaryAnnotator.CCExpansionDistance, self.base_raw.shape[0] - 1 - cc.max_y)

                        exp_h = cc.getHeight() + exp_min_y + exp_max_y
                        exp_w = cc.getWidth() + exp_min_x + exp_max_x

                        expansion = np.zeros((exp_h, exp_w), np.uint8)
                        expansion[exp_min_y:exp_h - exp_max_y, exp_min_x:exp_w - exp_max_x] = cc.img.copy()

                        dil_kernel = np.ones((1 + exp_min_y + exp_max_y, 1 + exp_min_x + exp_max_x), dtype=np.uint8)
                        dilated = cv2.dilate(expansion, np.array(dil_kernel, dtype=np.uint8))
                        expansion = dilated - expansion

                        abs_exp_min_x = cc.min_x - exp_min_x
                        abs_exp_max_x = cc.max_x + exp_max_x + 1
                        abs_exp_min_y = cc.min_y - exp_min_y
                        abs_exp_max_y = cc.max_y + exp_max_y + 1

                        expanded_cut = self.base_gray[abs_exp_min_y:abs_exp_max_y, abs_exp_min_x:abs_exp_max_x, 0].copy()
                        expanded_cut = np.abs(avg_cc_luminosity - expanded_cut.astype(np.float64))

                        expansion_pixels = []
                        for y in range(expanded_cut.shape[0]):
                            for x in range(expanded_cut.shape[1]):
                                if expansion[y, x] > 0:
                                    expansion_pixels.append((expanded_cut[y, x], abs_exp_min_y + y, abs_exp_min_x + x))

                        self.CC_expansion_pixels = sorted(expansion_pixels)
                        self.count_expand = 0

                        self.expansion_scroll.reset(0, len(self.CC_expansion_pixels), 0, 1)

                        self.set_editor_mode(GTPixelBinaryAnnotator.ModeGrowCC_Growing)
                        # force one view
                        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
                        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
                        self.update_current_view()
                        break

        elif button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Select:
             for cc in self.base_ccs:
                if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                    rel_offset_x = click_x - cc.min_x
                    rel_offset_y = click_y - cc.min_y

                    if cc.img[rel_offset_y, rel_offset_x] > 0:
                        self.selected_CC = cc
                        self.pre_edition_binary = self.base_binary.copy()

                        # compute CC stats ...
                        cc_cut = self.base_gray[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1, 0].copy()
                        cc_mask = cc.img > 0
                        avg_cc_luminosity = cc_cut[cc_mask].mean()

                        erode_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                        exp_img = np.zeros((cc.img.shape[0] + 2, cc.img.shape[1] + 2),  dtype=np.uint8)
                        exp_img[1:1 + cc.img.shape[0], 1:1 + cc.img.shape[1]] = cc.img.copy()
                        shrinked = cv2.erode(exp_img, erode_kernel)
                        reduction = exp_img - shrinked
                        reduction = reduction[1:1 + cc.img.shape[0], 1:1 + cc.img.shape[1]]

                        reduction_pixels = []
                        for y in range(cc_cut.shape[0]):
                            for x in range(cc_cut.shape[1]):
                                if reduction[y, x] > 0:
                                    reduction_pixels.append((abs(avg_cc_luminosity - cc_cut[y, x]), cc.min_y + y, cc.min_x + x))

                        self.CC_shrinking_pixels = sorted(reduction_pixels, reverse=True)
                        self.count_shrink = 0

                        self.expansion_scroll.reset(0, len(self.CC_shrinking_pixels), 0, 1)
                        self.set_editor_mode(GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

                        # force one view
                        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
                        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
                        self.update_current_view()
                        break

    def img_mouse_motion(self, img_object, pos, rel, buttons):
        if buttons[0] > 0 and self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:

            scaled_x, scaled_y = pos
            move_x = int(scaled_x / self.view_scale)
            move_y = int(scaled_y / self.view_scale)

            if move_x < 0 or move_y < 0 or move_x >= self.base_raw.shape[1] or move_y >= self.base_raw.shape[0]:
                # out of boundaries
                return

            if (move_x != self.last_motion_set_x or move_y != self.last_motion_set_y) and (self.base_binary[move_y, move_x] != self.last_motion_polarity):
                # invert pixel ...
                self.base_binary[move_y, move_x] = 255 - self.base_binary[move_y, move_x]
                if self.base_binary[move_y, move_x] > 0:
                    self.base_cc_image[move_y, move_x, :] = 0
                else:
                    self.base_cc_image[move_y, move_x, 0] = 1
                    self.base_cc_image[move_y, move_x, 1] = 0
                    self.base_cc_image[move_y, move_x, 2] = 0

                self.last_motion_set_x = move_x
                self.last_motion_set_y = move_y

                self.update_current_view(False, (move_x - 1, move_x + 1, move_y - 1, move_y + 1))
        else:
            self.last_motion_set_x = None
            self.last_motion_set_y = None
            self.last_motion_polarity = None

    def btn_confirm_accept_click(self, button):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeConfirmCancel:
            if self.finished_callback is not None:
                # accept ...
                self.finished_callback(False, None)

            self.return_screen = self.parent_screen

    def btn_confirm_cancel_click(self, button):
        # simply return to navigation mode mode
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)

    def btn_edition_expand_click(self, button):
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeGrowCC_Select)

    def btn_edition_shrink_click(self, button):
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeShrinkCC_Select)

    def btn_expand_accept_click(self, button):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
            # apply changes
            for value, y, x in self.CC_expansion_pixels[:self.count_expand]:
                # mark as content ...
                self.base_binary[y, x] = 0

            self.CC_expansion_pixels = None
        elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
            # apply changes
            for value, y, x in self.CC_shrinking_pixels[:self.count_shrink]:
                # mark as background ...
                self.base_binary[y, x] = 255

            self.CC_shrinking_pixels = None

        # clear selection
        self.selected_CC = None

        # finish edition
        self.btn_edition_stop_click(self.btn_edition_stop)
        self.update_current_view()

    def btn_expand_cancel_click(self, button):
        # clear selection
        self.selected_CC = None
        self.CC_expansion_pixels = None

        # go back to navigation mode
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)
        self.update_current_view()

    def expansion_scroll_change(self, scroll):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
            self.count_expand = scroll.value
            self.lbl_cc_expansion.set_text("Expand CC, Pixels = " + str(self.count_expand))
        elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
            self.count_shrink = scroll.value
            self.lbl_cc_expansion.set_text("Shrinking CC, Pixels = " + str(self.count_shrink))

        start_x = self.selected_CC.min_x - GTPixelBinaryAnnotator.CCExpansionDistance
        end_x = self.selected_CC.max_x + GTPixelBinaryAnnotator.CCExpansionDistance
        start_y = self.selected_CC.min_y - GTPixelBinaryAnnotator.CCExpansionDistance
        end_y = self.selected_CC.max_y + GTPixelBinaryAnnotator.CCExpansionDistance

        self.update_current_view(False, (start_x, end_x, start_y, end_y))
