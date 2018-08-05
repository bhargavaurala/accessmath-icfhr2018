
import cv2
import math
import time
import numpy as np

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from AccessMath.interface.controls.screen import Screen
from AccessMath.interface.controls.screen_label import ScreenLabel
from AccessMath.interface.controls.screen_button import ScreenButton
from AccessMath.interface.controls.screen_image import ScreenImage
from AccessMath.interface.controls.screen_container import ScreenContainer
from AccessMath.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll

import scipy.ndimage.measurements as sci_mes
from scipy import stats

class GTBinaryAnnotator(Screen):
    ModeEdition = 1
    ModeAddPoint = 2
    ModeConfirmExit = 3
    ModeConfirmCancel = 4

    TimeMergeThresholdChange = 3.0

    HighSTDevMaxOtsuDensity = 0.15 # assumed normal regions
    LowSTDevMaxOtsuDensity = 0.001 # assumed background regions
    LocalLowGraySTDev = 7.5

    def __init__(self, size, raw_input, partial_gt, initial_K, parent_screen=None):
        Screen.__init__(self, "Ground Truth Binary Annotation Interface", size)

        self.small_mode = self.height < 800

        # pre-filter raw input?
        #raw_input = cv2.bilateralFilter(raw_input, 20, 50.0, 1.0)
        #print("AQUI?")

        self.base_img_raw = np.zeros(raw_input.shape, raw_input.dtype)
        # switch color channels from RGB to BGR ...
        self.base_img_raw[:, :, 0] = raw_input[:, :, 2].copy()
        self.base_img_raw[:, :, 1] = raw_input[:, :, 1].copy()
        self.base_img_raw[:, :, 2] = raw_input[:, :, 0].copy()

        self.base_img_gray = cv2.cvtColor(raw_input, cv2.COLOR_RGB2GRAY)

        #self.base_img_smooth =
        #for sigmaSpace in [10]:
        #    for sigmaColor in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #        cv2.imshow("Combination: Space = " + str(sigmaSpace) + ", color = " + str(sigmaColor) , cv2.bilateralFilter(raw_input, -1, sigmaColor, sigmaSpace))

        # cv2.waitKey()

        self.base_img_ccs = None
        self.base_img_ccs_count = None
        self.base_img_ccs_sizes = None

        # un-comment to use a CLAHE version of the image for thresholding
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 5))
        # self.base_img_gray = clahe.apply(self.base_img_gray)

        min_gray = self.base_img_gray.min()
        max_gray = self.base_img_gray.max()
        scale = 255.0 / (max_gray - min_gray)
        self.base_img_enhanced_gray = ((self.base_img_gray - min_gray) * scale).astype(np.uint8)

        self.input_partial_gt = partial_gt

        self.output_gt_filename = None
        self.output_bin_filename = None

        self.parent_screen = parent_screen
        self.finished_callback = None

        self.point_grid_rows = 1
        self.point_grid_cols = 1

        # self.point_grid_offset = 0
        self.point_grid_noise = 0

        self.smoothing_sigma_color = 0
        self.smoothing_sigma_space = 1

        if partial_gt is None:
            point_count, partial_gt = self.generate_initial_gt_adaptive_otsu(raw_input, self.point_grid_rows, self.point_grid_cols)


        # defaults
        # ... edition ...
        self.selection_range = 5
        self.selection_multi = False
        self.selected_points = []  # nothing selected
        self.editor_mode = GTBinaryAnnotator.ModeEdition
        self.undo_stack = []
        self.redo_stack = []

        # ... view parameters ...
        self.current_view = 1               # Raw view
        self.current_scale = 1.0            # No zoom
        self.show_labeled_points = True     # show points
        self.max_scale = None               # assume unknown

        # algorithm elements (caches)
        self.image_K = initial_K
        self.image_height = raw_input.shape[0]
        self.image_width = raw_input.shape[1]

        y_coords = np.tile(np.arange(self.image_height), self.image_width)
        x_coords = np.repeat(np.arange(self.image_width), self.image_height)
        self.image_all_points = np.transpose([y_coords, x_coords])

        self.image_rgb_scale = 16.0
        yx_coords_y = np.repeat(np.arange(self.image_height), self.image_width)
        yx_coords_x = np.tile(np.arange(self.image_width), self.image_height)
        all_colors = self.base_img_raw.reshape((self.image_height * self.image_width, 3)) * self.image_rgb_scale
        self.image_yxbgr_points = np.hstack((np.transpose([yx_coords_y, yx_coords_x]), all_colors))

        # Initially labeled points
        self.image_labeled_mask = np.nonzero(partial_gt[:, :, 1])
        self.image_labeled_thresholds = partial_gt[self.image_labeled_mask]
        self.image_labeled_thresholds = self.image_labeled_thresholds[:, 0]
        self.image_labeled_points = np.transpose(self.image_labeled_mask)

        # cached images
        self.cache_knn_weights = None
        self.cache_knn_indices = None
        self.base_img_thresholds = None
        self.base_img_binary = None
        self.base_img_combined = None

        self.update_base_images(0)

        self.view_raw_normal = {self.current_scale: self.base_img_raw}
        self.view_gray_normal = {self.current_scale: self.base_img_gray}
        self.view_binary_normal = {self.current_scale: self.base_img_binary}
        self.view_thresholds_normal = {self.current_scale: self.base_img_thresholds}
        self.view_combined_normal = {self.current_scale: self.base_img_combined}

        self.view_raw_marked = {}
        self.view_gray_marked = {}
        self.view_binary_marked = {}
        self.view_thresholds_marked = {}
        self.view_combined_marked = {}

        self.update_marked_view_images(0)

        # USER INTERFACE .....
        general_background = (80, 80, 95)
        text_color = (255, 255, 255)
        button_text_color = (192, 255, 128)
        button_back_color = (64, 64, 64)
        font_size = 18 if self.small_mode else 21
        short_gap = 5 if self.small_mode else 10
        mid_gap = 10 if self.small_mode else 20

        self.elements.back_color = general_background

        # Main panel with view control buttons
        view_button_height = 125 if self.small_mode else 170
        self.container_view_buttons = ScreenContainer("container_main_buttons", (300, view_button_height), back_color=general_background)
        self.container_view_buttons.position = (self.width - self.container_view_buttons.width - 10, 10)
        self.elements.append(self.container_view_buttons)

        button_width = 190
        button_left = (self.container_view_buttons.width - button_width) / 2

        button_2_width = 120
        button_2_left = int(self.container_view_buttons.width * 0.25) - button_2_width / 2
        button_2_right = int(self.container_view_buttons.width * 0.75) - button_2_width / 2

        button_3_width = 90
        button_3_left = 10
        button_3_middle = (self.container_view_buttons.width - button_3_width) / 2
        button_3_right = self.container_view_buttons.width - button_3_width - 10

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", font_size, 290, 1)
        self.lbl_zoom.position = (5, 5)
        self.lbl_zoom.set_background(general_background)
        self.lbl_zoom.set_color(text_color)

        if not self.small_mode:
            self.container_view_buttons.append(self.lbl_zoom)
            zoom_buttons_top = self.lbl_zoom.get_bottom() + 10
        else:
            zoom_buttons_top = 5

        self.btn_zoom_reduce = ScreenButton("btn_zoom_reduce", "[ - ]", font_size, button_3_width)
        self.btn_zoom_reduce.set_colors(button_text_color, button_back_color)
        self.btn_zoom_reduce.position = (button_3_left, zoom_buttons_top)
        self.btn_zoom_reduce.click_callback = self.btn_zoom_reduce_click
        self.container_view_buttons.append(self.btn_zoom_reduce)

        self.btn_zoom_increase = ScreenButton("btn_zoom_increase", "[ + ]", font_size, button_3_width)
        self.btn_zoom_increase.set_colors(button_text_color, button_back_color)
        self.btn_zoom_increase.position = (button_3_right, zoom_buttons_top)
        self.btn_zoom_increase.click_callback = self.btn_zoom_increase_click
        self.container_view_buttons.append(self.btn_zoom_increase)

        self.btn_zoom_zero = ScreenButton("btn_zoom_zero", "100%", font_size, button_3_width)
        self.btn_zoom_zero.set_colors(button_text_color, button_back_color)
        self.btn_zoom_zero.position = (button_3_middle, zoom_buttons_top)
        self.btn_zoom_zero.click_callback = self.btn_zoom_zero_click
        self.container_view_buttons.append(self.btn_zoom_zero)

        self.lbl_views = ScreenLabel("lbl_views", "Views", font_size, button_3_width, 1)
        self.lbl_views.position = (button_3_left, self.btn_zoom_zero.get_bottom() + mid_gap + 10)
        self.lbl_views.set_background(general_background)
        self.lbl_views.set_color(text_color)
        self.container_view_buttons.append(self.lbl_views)

        self.btn_view_raw = ScreenButton("btn_view_raw", "Raw", font_size, button_3_width)
        self.btn_view_raw.set_colors(button_text_color, button_back_color)
        self.btn_view_raw.position = (button_3_middle, self.btn_zoom_zero.get_bottom() + mid_gap)
        self.btn_view_raw.click_callback = self.btn_view_raw_click
        self.container_view_buttons.append(self.btn_view_raw)

        self.btn_view_gray = ScreenButton("btn_view_gray", "Gray", font_size, button_3_width)
        self.btn_view_gray.set_colors(button_text_color, button_back_color)
        self.btn_view_gray.position = (button_3_right, self.btn_zoom_zero.get_bottom() + mid_gap)
        self.btn_view_gray.click_callback = self.btn_view_gray_click
        self.container_view_buttons.append(self.btn_view_gray)

        self.btn_view_binary = ScreenButton("btn_view_binary", "Binary", font_size, button_3_width)
        self.btn_view_binary.set_colors(button_text_color, button_back_color)
        self.btn_view_binary.position = (button_3_left, self.btn_view_gray.get_bottom() + short_gap)
        self.btn_view_binary.click_callback = self.btn_view_binary_click
        self.container_view_buttons.append(self.btn_view_binary)

        self.btn_view_labels = ScreenButton("btn_view_labels", "T. Map", font_size, button_3_width)
        self.btn_view_labels.set_colors(button_text_color, button_back_color)
        self.btn_view_labels.position = (button_3_middle, self.btn_view_gray.get_bottom() + short_gap)
        self.btn_view_labels.click_callback = self.btn_view_labels_click
        self.container_view_buttons.append(self.btn_view_labels)

        self.btn_view_combined = ScreenButton("btn_view_combined", "Combo", font_size, button_3_width)
        self.btn_view_combined.set_colors(button_text_color, button_back_color)
        self.btn_view_combined.position = (button_3_right, self.btn_view_gray.get_bottom() + short_gap)
        self.btn_view_combined.click_callback = self.btn_view_combined_click
        self.container_view_buttons.append(self.btn_view_combined)

        # Panel with point option buttons (Count, Add, Remove)
        self.container_point_buttons = ScreenContainer("container_point_buttons", (300, 115), back_color=general_background)
        self.container_point_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.container_point_buttons.visible = True
        self.elements.append(self.container_point_buttons)

        # points count ....
        n_points = self.image_labeled_points.shape[0]
        self.lbl_points = ScreenLabel("lbl_points", "Labeled Points: " + str(n_points), font_size, 290, 1)
        self.lbl_points.position = (5, 5)
        self.lbl_points.set_background(general_background)
        self.lbl_points.set_color(text_color)
        self.container_point_buttons.append(self.lbl_points)

        self.btn_points_show = ScreenButton("btn_points_add", "Show", font_size, 90)
        self.btn_points_show.set_colors(button_text_color, button_back_color)
        self.btn_points_show.position = (10, self.lbl_points.get_bottom() + 10)
        self.btn_points_show.click_callback = self.btn_points_show_click
        self.btn_points_show.visible = not self.show_labeled_points
        self.container_point_buttons.append(self.btn_points_show)

        self.btn_points_hide = ScreenButton("btn_points_hide", "Hide", font_size, 90)
        self.btn_points_hide.set_colors(button_text_color, button_back_color)
        self.btn_points_hide.position = (10, self.lbl_points.get_bottom() + 10)
        self.btn_points_hide.click_callback = self.btn_points_hide_click
        self.btn_points_hide.visible = self.show_labeled_points
        self.container_point_buttons.append(self.btn_points_hide)

        self.btn_points_add = ScreenButton("btn_points_add", "Add", font_size, 90)
        self.btn_points_add.set_colors(button_text_color, button_back_color)
        self.btn_points_add.position = ((self.container_point_buttons.width - self.btn_points_add.width) / 2,
                                        self.lbl_points.get_bottom() + 10)
        self.btn_points_add.click_callback = self.btn_points_add_click
        self.container_point_buttons.append(self.btn_points_add)

        self.btn_points_del = ScreenButton("btn_points_del", "Delete", font_size, 90)
        self.btn_points_del.set_colors(button_text_color, button_back_color)
        self.btn_points_del.position = (self.container_point_buttons.width - self.btn_points_del.width - 10,
                                        self.lbl_points.get_bottom() + 10)
        self.btn_points_del.click_callback = self.btn_points_del_click
        self.btn_points_del.visible = False
        self.container_point_buttons.append(self.btn_points_del)

        # self.lbl_grid_rows = ScreenLabel("lbl_grid_rows", "Rows: " + str(self.point_grid_rows), font_size, 100, 0)
        # self.lbl_grid_rows.position = (5, self.lbl_grid_title.get_bottom() + 20)
        self.lbl_knn = ScreenLabel("lbl_knn", "K-NN: " + str(self.image_K), font_size, 100, 1)
        self.lbl_knn.position = (5, self.btn_points_add.get_bottom() + 20)
        self.lbl_knn.set_background(general_background)
        self.lbl_knn.set_color(text_color)
        self.container_point_buttons.append(self.lbl_knn)

        short_button_width = 90
        inc_dec_left = self.container_point_buttons.width / 2 - 35

        self.btn_knn_reduce_k = ScreenButton("btn_knn_reduce_k", "[ - ]", font_size, 70)
        self.btn_knn_reduce_k.set_colors(button_text_color, button_back_color)
        self.btn_knn_reduce_k.position = (inc_dec_left, self.btn_points_add.get_bottom() + 10)
        self.btn_knn_reduce_k.click_callback = self.btn_knn_reduce_k_click
        self.container_point_buttons.append(self.btn_knn_reduce_k)

        self.btn_knn_increase_k = ScreenButton("btn_knn_increase_k", "[ + ]", font_size, 70)
        self.btn_knn_increase_k.set_colors(button_text_color, button_back_color)
        self.btn_knn_increase_k.position = (self.btn_knn_reduce_k.get_right() + 20, self.btn_points_add.get_bottom() + 10)
        self.btn_knn_increase_k.click_callback = self.btn_knn_increase_k_click
        self.container_point_buttons.append(self.btn_knn_increase_k)

        grid_buttons_height = 280 if self.small_mode else 350
        self.container_grid_buttons = ScreenContainer("container_grid_buttons", (300, grid_buttons_height), back_color=general_background)
        self.container_grid_buttons.position = (self.container_point_buttons.get_left(), self.container_point_buttons.get_bottom() + 5)
        self.container_grid_buttons.visible = True
        self.elements.append(self.container_grid_buttons)

        self.lbl_grid_rows = ScreenLabel("lbl_grid_rows", "Rows: " + str(self.point_grid_rows), font_size, 100, 0)
        self.lbl_grid_rows.position = (5, 15)
        self.lbl_grid_rows.set_background(general_background)
        self.lbl_grid_rows.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_grid_rows)

        self.btn_grid_rows_dec = ScreenButton("btn_grid_rows_dec", "[ - ]", font_size, 70)
        self.btn_grid_rows_dec.set_colors(button_text_color, button_back_color)
        self.btn_grid_rows_dec.position = (self.container_grid_buttons.width / 2 - 35, 5)
        self.btn_grid_rows_dec.click_callback = self.btn_grid_rows_dec_click
        self.container_grid_buttons.append(self.btn_grid_rows_dec)

        self.btn_grid_rows_inc = ScreenButton("btn_grid_rows_inc", "[ + ]", font_size, 70)
        self.btn_grid_rows_inc.set_colors(button_text_color, button_back_color)
        self.btn_grid_rows_inc.position = (self.btn_grid_rows_dec.get_right() + 20, self.btn_grid_rows_dec.get_top())
        self.btn_grid_rows_inc.click_callback = self.btn_grid_rows_inc_click
        self.container_grid_buttons.append(self.btn_grid_rows_inc)

        self.lbl_grid_cols = ScreenLabel("lbl_grid_cols", "Cols: " + str(self.point_grid_cols), font_size, 100, 0)
        self.lbl_grid_cols.position = (5, self.btn_grid_rows_inc.get_bottom() + short_gap + 10)
        self.lbl_grid_cols.set_background(general_background)
        self.lbl_grid_cols.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_grid_cols)

        self.btn_grid_cols_dec = ScreenButton("btn_grid_cols_dec", "[ - ]", font_size, 70)
        self.btn_grid_cols_dec.set_colors(button_text_color, button_back_color)
        self.btn_grid_cols_dec.position = (self.container_grid_buttons.width / 2 - 35, self.btn_grid_rows_inc.get_bottom() + short_gap)
        self.btn_grid_cols_dec.click_callback = self.btn_grid_cols_dec_click
        self.container_grid_buttons.append(self.btn_grid_cols_dec)

        self.btn_grid_cols_inc = ScreenButton("btn_grid_cols_inc", "[ + ]", font_size, 70)
        self.btn_grid_cols_inc.set_colors(button_text_color, button_back_color)
        self.btn_grid_cols_inc.position = (self.btn_grid_cols_dec.get_right() + 20, self.btn_grid_cols_dec.get_top())
        self.btn_grid_cols_inc.click_callback = self.btn_grid_cols_inc_click
        self.container_grid_buttons.append(self.btn_grid_cols_inc)

        self.btn_grid_update = ScreenButton("btn_grid_update", "Set Point Grid", font_size, button_width)
        self.btn_grid_update.set_colors(button_text_color, button_back_color)
        self.btn_grid_update.position = (button_left, self.btn_grid_cols_inc.get_bottom() + short_gap)
        self.btn_grid_update.click_callback = self.btn_grid_update_click
        self.container_grid_buttons.append(self.btn_grid_update)

        self.lbl_grid_offset = ScreenLabel("lbl_grid_noise", "T. Offset: ", font_size, 100, 0)
        self.lbl_grid_offset.position = (5, self.btn_grid_update.get_bottom() + short_gap + 10)
        self.lbl_grid_offset.set_background(general_background)
        self.lbl_grid_offset.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_grid_offset)

        self.btn_grid_offset_dec = ScreenButton("btn_grid_noise_dec", "[ - ]", font_size, 70)
        self.btn_grid_offset_dec.set_colors(button_text_color, button_back_color)
        self.btn_grid_offset_dec.position = (self.container_grid_buttons.width / 2 - 35, self.btn_grid_update.get_bottom() + short_gap)
        self.btn_grid_offset_dec.click_callback = self.btn_grid_offset_dec_click
        self.container_grid_buttons.append(self.btn_grid_offset_dec)

        self.btn_grid_offset_inc = ScreenButton("btn_grid_offset_inc", "[ + ]", font_size, 70)
        self.btn_grid_offset_inc.set_colors(button_text_color, button_back_color)
        self.btn_grid_offset_inc.position = (self.btn_grid_offset_dec.get_right() + 20, self.btn_grid_offset_dec.get_top())
        self.btn_grid_offset_inc.click_callback = self.btn_grid_offset_inc_click
        self.container_grid_buttons.append(self.btn_grid_offset_inc)

        self.lbl_grid_noise = ScreenLabel("lbl_grid_noise", "Min Size: " + str(self.point_grid_noise), font_size, 100, 0)
        self.lbl_grid_noise.position = (5, self.btn_grid_offset_inc.get_bottom() + short_gap + 10)
        self.lbl_grid_noise.set_background(general_background)
        self.lbl_grid_noise.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_grid_noise)

        self.btn_grid_noise_dec = ScreenButton("btn_grid_noise_dec", "[ - ]", font_size, 70)
        self.btn_grid_noise_dec.set_colors(button_text_color, button_back_color)
        self.btn_grid_noise_dec.position = (self.container_grid_buttons.width / 2 - 35, self.btn_grid_offset_inc.get_bottom() + short_gap)
        self.btn_grid_noise_dec.click_callback = self.btn_grid_noise_dec_click
        self.container_grid_buttons.append(self.btn_grid_noise_dec)

        self.btn_grid_noise_inc = ScreenButton("btn_grid_noise_inc", "[ + ]", font_size, 70)
        self.btn_grid_noise_inc.set_colors(button_text_color, button_back_color)
        self.btn_grid_noise_inc.position = (self.btn_grid_noise_dec.get_right() + 20, self.btn_grid_noise_dec.get_top())
        self.btn_grid_noise_inc.click_callback = self.btn_grid_noise_inc_click
        self.container_grid_buttons.append(self.btn_grid_noise_inc)

        self.lbl_smoothing_color = ScreenLabel("lbl_smoothing_color", "Smoothing Sigma Color: " + str(self.smoothing_sigma_color), font_size, 290, 1)
        self.lbl_smoothing_color.position = (5, self.btn_grid_noise_inc.get_bottom() + short_gap)
        self.lbl_smoothing_color.set_background(general_background)
        self.lbl_smoothing_color.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_smoothing_color)

        self.scroll_smoothing_color = ScreenHorizontalScroll("scroll_smoothing_color", 0, 100, 0, 10)
        self.scroll_smoothing_color.position = (5, self.lbl_smoothing_color.get_bottom() + short_gap)
        self.scroll_smoothing_color.width = 290
        self.scroll_smoothing_color.scroll_callback = self.scroll_smoothing_color_change
        self.container_grid_buttons.append(self.scroll_smoothing_color)

        self.lbl_smoothing_space = ScreenLabel("lbl_smoothing_space", "Smoothing Sigma Space: " + str(self.smoothing_sigma_space), font_size, 290, 1)
        self.lbl_smoothing_space.position = (5, self.scroll_smoothing_color.get_bottom() + short_gap)
        self.lbl_smoothing_space.set_background(general_background)
        self.lbl_smoothing_space.set_color(text_color)
        self.container_grid_buttons.append(self.lbl_smoothing_space)

        self.scroll_smoothing_space = ScreenHorizontalScroll("scroll_smoothing_space", 1, 100, 0, 10)
        self.scroll_smoothing_space.position = (5, self.lbl_smoothing_space.get_bottom() + short_gap)
        self.scroll_smoothing_space.width = 290
        self.scroll_smoothing_space.scroll_callback = self.scroll_smoothing_space_change
        self.container_grid_buttons.append(self.scroll_smoothing_space)

        # Panel with confirmation buttons (Message, Accept, Cancel)
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (300, 70), back_color=general_background)
        self.container_confirm_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 10)
        self.elements.append(self.container_confirm_buttons)

        self.lbl_confirm = ScreenLabel("lbl_confirm", "Exit without saving?", font_size, 290, 1)
        self.lbl_confirm.position = (5, 5)
        self.lbl_confirm.set_background(general_background)
        self.lbl_confirm.set_color(text_color)
        self.container_confirm_buttons.append(self.lbl_confirm)

        self.btn_confirm_accept = ScreenButton("btn_confirm_accept", "Accept", font_size, 130)
        self.btn_confirm_accept.set_colors(button_text_color, button_back_color)
        self.btn_confirm_accept.position = (10, self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_accept.click_callback = self.btn_confirm_accept_click
        self.container_confirm_buttons.append(self.btn_confirm_accept)

        self.btn_confirm_cancel = ScreenButton("btn_confirm_cancel", "Cancel", font_size, 130)
        self.btn_confirm_cancel.set_colors(button_text_color, button_back_color)
        self.btn_confirm_cancel.position = (self.container_confirm_buttons.width - self.btn_confirm_cancel.width - 10,
                                            self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_cancel.click_callback = self.btn_confirm_cancel_click
        self.container_confirm_buttons.append(self.btn_confirm_cancel)
        self.container_confirm_buttons.visible = False

        # Panel with state buttons (Undo, Redo, Save)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (300, 120), general_background)
        self.container_state_buttons.position = (self.container_view_buttons.get_left(), self.container_grid_buttons.get_bottom() + 10)
        self.elements.append(self.container_state_buttons)

        self.btn_undo = ScreenButton("btn_undo", "Undo", font_size, button_3_width)
        self.btn_undo.set_colors(button_text_color, button_back_color)
        self.btn_undo.position = (button_3_left, 5)
        self.btn_undo.click_callback = self.btn_undo_click
        self.container_state_buttons.append(self.btn_undo)

        self.btn_redo = ScreenButton("btn_redo", "Redo", font_size, button_3_width)
        self.btn_redo.set_colors(button_text_color, button_back_color)
        self.btn_redo.position = (button_3_middle, 5)
        self.btn_redo.click_callback = self.btn_redo_click
        self.container_state_buttons.append(self.btn_redo)

        if self.parent_screen is None:
            # Stand-alone mode

            # Add Save button
            self.btn_save = ScreenButton("btn_save", "Save", font_size, button_3_width)
            self.btn_save.set_colors(button_text_color, button_back_color)
            self.btn_save.position = (button_3_right, 5)
            self.btn_save.click_callback = self.btn_save_click
            self.container_state_buttons.append(self.btn_save)

            # Add exit button
            self.btn_exit = ScreenButton("btn_exit", "Exit", font_size, button_width)
            self.btn_exit.set_colors(button_text_color, button_back_color)
            self.btn_exit.position = (button_left, self.btn_undo.get_bottom() + 30)
            self.btn_exit.click_callback = self.btn_exit_click
            self.container_state_buttons.append(self.btn_exit)
        else:
            # Secondary screen mode
            # Add Cancel Button
            self.btn_return_cancel = ScreenButton("btn_return_cancel", "Cancel", font_size, button_2_width)
            self.btn_return_cancel.set_colors(button_text_color, button_back_color)
            self.btn_return_cancel.position = (int(self.container_state_buttons.width * 0.25) - self.btn_return_cancel.width / 2,
                                               self.btn_undo.get_bottom() + 30)
            self.btn_return_cancel.click_callback = self.btn_return_cancel_click
            self.container_state_buttons.append(self.btn_return_cancel)

            # Add Accept Button
            self.btn_return_accept = ScreenButton("btn_return_accept", "Accept", font_size, button_2_width)
            self.btn_return_accept.set_colors(button_text_color, button_back_color)
            self.btn_return_accept.position = (button_2_right, self.btn_undo.get_bottom() + 30)
            self.btn_return_accept.click_callback = self.btn_return_accept_click
            self.container_state_buttons.append(self.btn_return_accept)

        # Image panel ...
        image_width = self.width - self.container_view_buttons.width - 30
        image_height = self.height - 70
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, 10)
        self.elements.append(self.container_images)

        # Threshold edit panel
        self.container_threshold = ScreenContainer("container_threshold", (self.container_images.width - 200, 30),
                                                back_color=general_background)
        self.container_threshold.position = (10, self.container_images.get_bottom() + 10)
        self.elements.append(self.container_threshold)
        self.container_threshold.visible = False

        self.lbl_threshold = ScreenLabel("lbl_threshold", "Threshold: 255", font_size, centered=0)
        self.lbl_threshold.position = (5, 5)
        self.lbl_threshold.set_color(text_color)
        self.lbl_threshold.set_background(general_background)
        self.container_threshold.append(self.lbl_threshold)

        self.threshold_scroll = ScreenHorizontalScroll("threshold_scroll", 0, 255, 128, 10)
        self.threshold_scroll.position = (self.lbl_threshold.get_right() + 10, 5)
        self.threshold_scroll.width = self.container_threshold.width - self.threshold_scroll.get_left() - 10
        self.threshold_scroll.scroll_callback = self.threshold_scroll_change
        self.container_threshold.append(self.threshold_scroll)

        # multi-selection options
        self.btn_multi_select_start = ScreenButton("btn_multi_select_start", "Multiple Selection", font_size, 180)
        self.btn_multi_select_start.set_colors(button_text_color, button_back_color)
        self.btn_multi_select_start.position = (self.container_threshold.get_right() + 10, self.container_threshold.get_top())
        self.btn_multi_select_start.click_callback = self.btn_multi_select_start_click
        self.elements.append(self.btn_multi_select_start)

        self.btn_multi_select_clear = ScreenButton("btn_multi_select_clear", "Clear Selection", font_size, 180)
        self.btn_multi_select_clear.set_colors(button_text_color, button_back_color)
        self.btn_multi_select_clear.position = (self.container_threshold.get_right() + 10, self.container_threshold.get_top())
        self.btn_multi_select_clear.click_callback = self.btn_multi_select_clear_click
        self.btn_multi_select_clear.visible = False
        self.elements.append(self.btn_multi_select_clear)

        # ... image objects ...
        self.img_raw = ScreenImage("img_raw", raw_input, 0, 0, True, cv2.INTER_NEAREST)
        self.img_raw.position = (0, 0)
        self.img_raw.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_raw)

        self.img_binary = ScreenImage("img_binary", raw_input, 0, 0, True, cv2.INTER_NEAREST)
        self.img_binary.position = (0, 0)
        self.img_binary.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_binary)
        self.img_binary.visible = False

        self.img_gray = ScreenImage("img_gray", raw_input, 0, 0, True, cv2.INTER_NEAREST)
        self.img_gray.position = (0, 0)
        self.img_gray.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_gray)
        self.img_gray.visible = False

        self.img_thresholds = ScreenImage("img_thresholds", raw_input, 0, 0, True, cv2.INTER_NEAREST)
        self.img_thresholds.position = (0, 0)
        self.img_thresholds.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_thresholds)
        self.img_thresholds.visible = False

        self.img_combined = ScreenImage("img_combined", raw_input, 0, 0, True, cv2.INTER_NEAREST)
        self.img_combined.position = (0, 0)
        self.img_combined.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_combined)
        self.img_combined.visible = False

        # ... set up the initial view ...
        self.update_view_image(True)


    def set_output_files(self, output_gt_filename, output_bin_filename):
        self.output_gt_filename = output_gt_filename
        self.output_bin_filename = output_bin_filename

    def btn_zoom_reduce_click(self, button):
        if self.current_scale > 1.0:
            self.change_zoom(self.current_scale, self.current_scale - 1.0)

    def btn_zoom_increase_click(self, button):
        if self.max_scale is None or self.current_scale < self.max_scale:
            # try increased resolution/scale ....
            previous_scale = self.current_scale
            try:
                self.change_zoom(previous_scale, self.current_scale + 1.0)

                if self.current_scale > 15.0:
                    # no need to increase from this point ...
                    self.max_scale = self.current_scale
            except Exception as E:
                # go back to previous resolution
                self.change_zoom(self.current_scale, previous_scale)
                # set maximum resolution
                self.max_scale = self.current_scale

                print(E)

    def btn_zoom_zero_click(self, button):
        old_scale = self.current_scale
        self.change_zoom(old_scale, 1.0)

    def change_zoom(self, old_zoom, new_zoom):
        # keep previous offsets ...
        scroll_offset_y = self.container_images.v_scroll.value if self.container_images.v_scroll.active else 0
        scroll_offset_x = self.container_images.h_scroll.value if self.container_images.h_scroll.active else 0

        prev_center_y = scroll_offset_y + self.container_images.height / 2
        prev_center_x = scroll_offset_x + self.container_images.width / 2

        # compute new scroll bar offsets
        new_off_y = (prev_center_y / old_zoom) * new_zoom - self.container_images.height / 2
        new_off_x = (prev_center_x / old_zoom) * new_zoom - self.container_images.width / 2

        # new zoomed size
        self.current_scale = new_zoom
        #print((old_zoom, new_zoom))

        # update image ...
        self.update_view_image(True)

        # set offsets
        if self.container_images.v_scroll.active and 0 <= new_off_y <= self.container_images.v_scroll.max:
            self.container_images.v_scroll.value = new_off_y
        if self.container_images.h_scroll.active and 0 <= new_off_x <= self.container_images.h_scroll.max:
            self.container_images.h_scroll.value = new_off_x

        self.lbl_zoom.set_text("Zoom: " + str(int(new_zoom * 100.0)) + "%")

    def img_mouse_down(self, img_object, pos, button):
        if button == 1:
            # ... first, get click location on original image space
            scaled_x, scaled_y = pos
            click_x = int(scaled_x / self.current_scale)
            click_y = int(scaled_y / self.current_scale)

            if self.editor_mode == GTBinaryAnnotator.ModeEdition:
                # check location against current points ...
                n_points = self.image_labeled_points.shape[0]
                point_clicked = None
                for p_idx in range(n_points):
                    point_y, point_x = self.image_labeled_points[p_idx]

                    distance = math.sqrt(math.pow(point_x - click_x, 2) + math.pow(point_y - click_y, 2))
                    if distance * self.current_scale < self.selection_range:
                        # stop searching ...
                        point_clicked = p_idx
                        break

                update = False
                if point_clicked is None:
                    if not self.selection_multi:
                        # not multi-selection, nothing clicked, clear selection
                        self.selected_points = []
                        update = True
                else:

                    if point_clicked not in self.selected_points:
                        # either add point to selection or change point selected
                        if self.selection_multi:
                            self.selected_points.append(point_clicked)
                        else:
                            self.selected_points = [point_clicked]
                        update = True

                if update:
                    self.update_view_image(False)
                    self.update_selected_options()

            if self.editor_mode == GTBinaryAnnotator.ModeAddPoint:
                n_points = self.image_labeled_points.shape[0]

                # threshold ....
                initial_threshold = self.base_img_thresholds[click_y, click_x]

                # add point
                self.add_point(click_x, click_y, initial_threshold, n_points, add_undo=True)

                # prepare interface to edit new point
                self.selected_points = [n_points]
                self.set_edition_mode(GTBinaryAnnotator.ModeEdition)
                self.update_selected_options()


    def btn_view_raw_click(self, button):
        self.change_view(1)

    def btn_view_gray_click(self, button):
        self.change_view(5)

    def btn_view_binary_click(self, button):
        self.change_view(2)

    def btn_view_labels_click(self, button):
        self.change_view(3)

    def btn_view_combined_click(self, button):
        self.change_view(4)

    def change_view(self, new_view):
        self.current_view = new_view
        self.update_view_image(False)

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False
        if to_undo["operation"] == "image_K_changed":
            # Restore old K, don't add to the list of operations to undo
            self.update_image_K(to_undo["old_K"], False)
            success = True
        elif to_undo["operation"] == "threshold_changed":
            # restore previous threshold value
            self.change_threshold_value(to_undo["point_idx"], to_undo["new_value"], to_undo["old_value"], False)
            success = True

        elif to_undo["operation"] == "point_deleted" or to_undo["operation"] == "grid_applied":
            # restore state of points ...
            self.image_labeled_mask = (to_undo["old_mask"][0].copy(), to_undo["old_mask"][1].copy())
            self.image_labeled_thresholds = to_undo["old_thresholds"].copy()
            self.image_labeled_points = to_undo["old_points"].copy()
            self.image_K = to_undo["old_K"]
            self.lbl_knn.set_text("K-NN: " + str(self.image_K))
            self.grid_changed_update(0, True)
            success = True

        elif to_undo["operation"] == "point_added":
            # delete point ...
            self.delete_points([to_undo["idx"]], False)
            success = True

        elif to_undo["operation"] == "offset_applied":
            # restore previous values ...
            self.image_labeled_thresholds = to_undo["old_values"]
            self.apply_offset_all_points(0, False)
            success = True

        else:
            raise Exception("Undo operation not implemented: " + str(to_undo))

        # ... removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # set edition mode by default
            self.set_edition_mode(GTBinaryAnnotator.ModeEdition)
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

            # copy last operation
        to_redo = self.redo_stack[-1]

        success = False
        if to_redo["operation"] == "image_K_changed":
            self.update_image_K(to_redo["new_K"], False)
            success = True
        elif to_redo["operation"] == "threshold_changed":
            # apply new threshold again
            self.change_threshold_value(to_redo["point_idx"], to_redo["old_value"], to_redo["new_value"], False)
            success = True
        elif to_redo["operation"] == "point_added":
            self.add_point(to_redo["x"], to_redo["y"], to_redo["threshold"], to_redo["idx"], to_redo["new_K"], False)
            success = True

        elif to_redo["operation"] == "offset_applied":
            # restore previous values ...
            self.image_labeled_thresholds = to_redo["new_values"]
            self.apply_offset_all_points(0, False)
            success = True

        elif to_redo["operation"] == "point_deleted" or to_redo["operation"] == "grid_applied":
            self.image_labeled_mask = (to_redo["new_mask"][0].copy(), to_redo["new_mask"][1].copy())
            self.image_labeled_thresholds = to_redo["new_thresholds"].copy()
            self.image_labeled_points = to_redo["new_points"].copy()
            self.image_K = to_redo["new_K"]
            self.lbl_knn.set_text("K-NN Threshold Interpolation K: " + str(self.image_K))
            self.grid_changed_update(0, True)
            success = True

        else:
            raise Exception("Undo operation not implemented: " + str(to_redo))

        if success:
            self.undo_stack.append(to_redo)
            # removing last operation
            del self.redo_stack[-1]

            # set edition mode by default
            self.set_edition_mode(GTBinaryAnnotator.ModeEdition)
        else:
            print("Action could not be re-done!")

    def btn_save_click(self, button):
        print("Not yet implemented")

    def btn_exit_click(self, button):
        if len(self.undo_stack) == 0:
            # just close, nothing to save ...
            self.return_screen = None
            print("APPLICATION FINISHED")
        else:
            self.set_edition_mode(GTBinaryAnnotator.ModeConfirmExit)

    def btn_points_add_click(self, button):
        self.set_edition_mode(GTBinaryAnnotator.ModeAddPoint)

    def btn_points_del_click(self, button):
        n_points = self.image_labeled_points.shape[0]

        if len(self.selected_points) < n_points:
            self.delete_points(self.selected_points, True)

    def btn_confirm_accept_click(self, button):
        if self.editor_mode == GTBinaryAnnotator.ModeConfirmExit:
            # confirmed to exit without saving changes
            print("Changes Discarded")
            print("APPLICATION FINISHED")
            self.return_screen = None
        elif self.editor_mode == GTBinaryAnnotator.ModeConfirmCancel:
            # return cancel ...
            if self.finished_callback is not None:
                # cancel ...
                self.finished_callback(False, None)
            self.return_screen = self.parent_screen

    def btn_confirm_cancel_click(self, button):
        if (self.editor_mode == GTBinaryAnnotator.ModeConfirmExit or
            self.editor_mode == GTBinaryAnnotator.ModeAddPoint or
            self.editor_mode == GTBinaryAnnotator.ModeConfirmCancel):
            # go back to edit mode
            self.set_edition_mode(GTBinaryAnnotator.ModeEdition)


    def add_point(self, x, y, initial_threshold, idx=None, new_K=None, add_undo=True):
        if idx is None:
            idx = self.image_labeled_points.shape[0]

        # create new structures adding the point
        new_labeled_points = np.vstack((self.image_labeled_points[:idx], np.array([y, x]),
                                        self.image_labeled_points[idx:]))
        new_labeled_thresholds = np.hstack((self.image_labeled_thresholds[:idx], np.array([initial_threshold]),
                                            self.image_labeled_thresholds[idx:]))

        old_mask_y, old_mask_x = self.image_labeled_mask
        new_mask_x = np.hstack((old_mask_x[:idx], np.array([x]), old_mask_x[idx:]))
        new_mask_y = np.hstack((old_mask_y[:idx], np.array([y]), old_mask_y[idx:]))
        new_labeled_mask = (new_mask_y, new_mask_x)

        # replace structures
        self.image_labeled_points = new_labeled_points
        self.image_labeled_thresholds = new_labeled_thresholds
        self.image_labeled_mask = new_labeled_mask

        # update K (if requested)
        old_k = self.image_K
        if new_K is not None and self.image_K != new_K:
            self.image_K = new_K
            self.lbl_knn.set_text("K-NN Threshold Interpolation K: " + str(self.image_K))

        # update view
        self.selected_points = []
        self.lbl_points.set_text("Labeled Points: " + str(self.image_labeled_points.shape[0]))
        self.update_base_images(0)
        self.clean_view_cache()
        self.update_view_image(False)

        # add to undo
        if add_undo:
            self.undo_stack.append({
                "operation": "point_added",
                "idx": idx,
                "x": x,
                "y": y,
                "old_K": old_k,
                "new_K": self.image_K,
                "threshold": initial_threshold,
            })

    def delete_points(self, indices, add_undo=True):
        # make copies
        prev_mask = (self.image_labeled_mask[0].copy(), self.image_labeled_mask[1].copy())
        prev_thresholds = self.image_labeled_thresholds.copy()
        prev_points = self.image_labeled_points.copy()

        new_labeled_mask = (self.image_labeled_mask[0].copy(), self.image_labeled_mask[1].copy())
        new_labeled_thresholds = self.image_labeled_thresholds.copy()
        new_labeled_points = self.image_labeled_points.copy()

        for idx in sorted(indices, reverse=True):
            # create new structures without the point
            if idx == new_labeled_points.shape[0] - 1:
                # special case: last point
                new_labeled_points = new_labeled_points[:idx]
                new_labeled_thresholds = new_labeled_thresholds[:idx]

                old_mask_y, old_mask_x = new_labeled_mask
                new_mask_x = old_mask_x[:idx]
                new_mask_y = old_mask_y[:idx]
                new_labeled_mask = (new_mask_y, new_mask_x)
            else:
                new_labeled_points = np.vstack((new_labeled_points[:idx], new_labeled_points[idx + 1:]))
                new_labeled_thresholds = np.hstack((new_labeled_thresholds[:idx], new_labeled_thresholds[idx + 1:]))

                old_mask_y, old_mask_x = new_labeled_mask
                new_mask_x = np.hstack((old_mask_x[:idx], old_mask_x[idx + 1:]))
                new_mask_y = np.hstack((old_mask_y[:idx], old_mask_y[idx + 1:]))
                new_labeled_mask = (new_mask_y, new_mask_x)

        # replace structures
        self.image_labeled_points = new_labeled_points
        self.image_labeled_thresholds = new_labeled_thresholds
        self.image_labeled_mask = new_labeled_mask

        # check K for consistency, can't be larger than total of points
        old_k = self.image_K
        n_points = self.image_labeled_points.shape[0]
        if self.image_K > n_points:
            # reduce K
            self.image_K = n_points
            self.lbl_knn.set_text("K-NN Threshold Interpolation K: " + str(self.image_K))

        new_k = self.image_K

        # update views
        self.grid_changed_update(0, True)

        # add to undo
        if add_undo:
            self.undo_stack.append({
                "operation": "point_deleted",
                "old_mask": prev_mask,
                "old_thresholds": prev_thresholds,
                "old_points": prev_points,
                "new_mask": (self.image_labeled_mask[0].copy(), self.image_labeled_mask[1].copy()),
                "new_thresholds": self.image_labeled_thresholds.copy(),
                "new_points": self.image_labeled_points.copy(),
                "old_K": old_k,
                "new_K": new_k,
            })


    def btn_knn_increase_k_click(self, button):
        n_points = self.image_labeled_points.shape[0]

        if self.image_K < n_points:
            self.update_image_K(self.image_K + 1)

    def btn_knn_reduce_k_click(self, button):
        if self.image_K > 1:
            self.update_image_K(self.image_K - 1)

    def update_image_K(self, new_K, add_to_undo_stack=True):
        previous_K = self.image_K
        self.image_K = new_K

        self.lbl_knn.set_text("K-NN Threshold Interpolation K: " + str(self.image_K))

        # update the base images
        self.update_base_images(0)

        # empty caches
        self.clean_view_cache()

        # udpate the current view
        self.update_view_image(False)

        # add to list of operations to Undo
        if add_to_undo_stack:
            self.undo_stack.append({
                "operation": "image_K_changed",
                "old_K" : previous_K,
                "new_K" : new_K,
            })

    def btn_points_show_click(self, button):
        self.change_labeled_point_visibilty(True)

    def btn_points_hide_click(self, button):
        self.change_labeled_point_visibilty(False)

    def change_labeled_point_visibilty(self, show):
        self.show_labeled_points = show
        self.btn_points_show.visible = not show
        self.btn_points_hide.visible = show

        self.update_view_image(False)

    def threshold_scroll_change(self, scroll):
        if len(self.selected_points) == 1:
            previous_value = self.image_labeled_thresholds[self.selected_points[0]]
            new_value = int(scroll.value)
            self.change_threshold_value(self.selected_points[0], previous_value, new_value)

    def change_threshold_value(self, point_idx, old_value, new_value, add_undo=True):
        # change ....
        self.image_labeled_thresholds[point_idx] = new_value

        # update base
        self.update_base_images(1)
        # update visuals
        self.clean_view_cache()
        self.update_view_image(False)

        self.lbl_threshold.set_text("Threshold: " + str(new_value))
        self.threshold_scroll.value = new_value

        if add_undo:
            # check if merge ....
            if (len(self.undo_stack) > 0 and self.undo_stack[-1]["operation"] == "threshold_changed" and
                self.undo_stack[-1]["point_idx"] == point_idx and
                time.time() - self.undo_stack[-1]["time"] < GTBinaryAnnotator.TimeMergeThresholdChange):
                # merge with last operation
                self.undo_stack[-1]["new_value"] = new_value
            else:
                # add new
                self.undo_stack.append({
                    "operation": "threshold_changed",
                    "time": time.time(),
                    "point_idx": point_idx,
                    "old_value": old_value,
                    "new_value": new_value,
                })

    def update_base_images(self, starting_step):
        if starting_step == -1:
            # pre-processing steps ...
            # recompute gray-scale image
            if self.smoothing_sigma_color > 0:
                color_source = cv2.bilateralFilter(self.base_img_raw, -1, self.smoothing_sigma_color, self.smoothing_sigma_space)
            else:
                color_source = self.base_img_raw

            self.base_img_gray = cv2.cvtColor(color_source, cv2.COLOR_RGB2GRAY)

        if starting_step == 0:
            # building distance cache from scratch
            # Required at the beginning and when a point has been added or deleted
            # generate a NN structure
            n_neighbors = NearestNeighbors(n_neighbors=self.image_K)
            n_neighbors.fit(self.image_labeled_points)

            # find closest labeled neighbors for each pixel
            distances, self.cache_knn_indices = n_neighbors.kneighbors(self.image_all_points)

            # computer inverse weights
            # first, replace zero distances with arbitrary value
            # note that this corresponds with labeled points
            distances[distances == 0.0] = 1.0

            # compute weights
            weights = 1.0 / distances
            norms = np.sum(weights, axis=1)
            norms = np.repeat(norms, self.image_K).reshape(self.image_height * self.image_width, self.image_K)
            weights /= norms

            self.cache_knn_weights = weights

        if starting_step <= 1:
            # update thresholds
            propagated_labels = self.image_labeled_thresholds[self.cache_knn_indices]

            self.base_img_thresholds = np.sum(self.cache_knn_weights * propagated_labels, axis=1).reshape(self.image_width, self.image_height)
            self.base_img_thresholds = np.transpose(self.base_img_thresholds).astype(np.uint8)

            # use true labels on labeled points
            self.base_img_thresholds[self.image_labeled_mask] = self.image_labeled_thresholds

        if starting_step <= 2:
            # update binary
            self.base_img_binary = np.zeros(self.base_img_raw.shape, np.uint8)
            binary_mask = self.base_img_gray > self.base_img_thresholds
            self.base_img_binary[binary_mask, 0] = 255
            self.base_img_binary[binary_mask, 1] = 255
            self.base_img_binary[binary_mask, 2] = 255

            #print("at here: " + str(self.base_img_binary.sum()))

            self.base_img_ccs = None
            self.base_img_ccs_count = None
            self.base_img_ccs_sizes = None

        if starting_step <= 3:
            # only execute this steps if noise filtering is active ...
            if self.point_grid_noise > 0:
                # if no CC cache has been computed (or it was out-dated)
                if self.base_img_ccs is None:
                    # label CCs
                    self.base_img_ccs, self.base_img_ccs_count = sci_mes.label(255 - self.base_img_binary[:, :, 0])

                    # get sizes
                    self.base_img_ccs_sizes = []
                    for idx in range(1, self.base_img_ccs_count + 1):
                        self.base_img_ccs_sizes.append((self.base_img_ccs == idx).sum())

                print("Before filtering CCs: " + str(self.base_img_ccs_count))

                cc_removed = 0
                filtered_ccs = self.base_img_ccs.copy()
                for idx in range(self.base_img_ccs_count):
                    if self.base_img_ccs_sizes[idx] <= self.point_grid_noise:
                        # remove CC
                        filtered_ccs[filtered_ccs == idx + 1] = 0
                        cc_removed += 1

                # update binary image
                binary_mask = filtered_ccs == 0
                self.base_img_binary[:, :, :] = 0
                self.base_img_binary[binary_mask, 0] = 255
                self.base_img_binary[binary_mask, 1] = 255
                self.base_img_binary[binary_mask, 2] = 255

                #print("at there!!: " + str(self.base_img_binary.sum()))

                print("After filtering CCs: " + str(self.base_img_ccs_count - cc_removed))

        if starting_step <= 4:
            # update combined
            inverse_binary_mask = self.base_img_binary[:, :, 0] == 0
            self.base_img_combined = np.zeros(self.base_img_raw.shape, np.uint8)
            self.base_img_combined[:, :, 0] = self.base_img_enhanced_gray.copy()
            self.base_img_combined[:, :, 1] = self.base_img_enhanced_gray.copy()
            self.base_img_combined[:, :, 2] = self.base_img_enhanced_gray.copy()

            self.base_img_combined[inverse_binary_mask, 0] = 1

    def create_normal_view_images(self, to_create):
        # create normal view images for current scale

        zoom_width = int(self.image_width * self.current_scale)
        zoom_height = int(self.image_height * self.current_scale)

        # 0 - creates all
        # 1 - only create raw
        if to_create == 0 or to_create == 1:
            if self.current_scale not in self.view_raw_normal:
                result = cv2.resize(self.base_img_raw, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
                self.view_raw_normal[self.current_scale] = result

        # 2 - only create binary
        if to_create == 0 or to_create == 2:
            if self.current_scale not in self.view_binary_normal:
                result = cv2.resize(self.base_img_binary, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
                self.view_binary_normal[self.current_scale] = result

        # 3 - only update thresholds
        if to_create == 0 or to_create == 3:
            if self.current_scale not in self.view_thresholds_normal:
                result = cv2.resize(self.base_img_thresholds, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
                self.view_thresholds_normal[self.current_scale] = result

        # 4 - only update combined
        if to_create == 0 or to_create == 4:
            if self.current_scale not in self.view_combined_normal:
                result = cv2.resize(self.base_img_combined, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
                self.view_combined_normal[self.current_scale] = result

        # 5 - only update gray scale
        if to_create == 0 or to_create == 5:
            if self.current_scale not in self.view_gray_normal:
                result = cv2.resize(self.base_img_gray, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
                self.view_gray_normal[self.current_scale] = result

    def mark_points_in_view(self, input_image, normal_color=(0, 128, 0), selected_color=(192, 128, 0)):
        # Draw all points
        n_points = self.image_labeled_points.shape[0]

        for idx in range(n_points):
            y, x = self.image_labeled_points[idx]

            color = selected_color if idx in self.selected_points else normal_color

            scaled_center = (int((x + 0.5) * self.current_scale), int((y + 0.5) * self.current_scale))

            cv2.circle(input_image, scaled_center, self.selection_range, color, -1)

    def update_marked_view_images(self, to_update):
        # note that this methods assumes that normal views images for current scale exists

        # 0 - updates all
        # 1 - only update raw
        if to_update == 0 or to_update == 1:
            # check if copy exists, if not create ...
            if self.current_scale not in self.view_raw_marked:
                self.view_raw_marked[self.current_scale] = self.view_raw_normal[self.current_scale].copy()

            self.mark_points_in_view(self.view_raw_marked[self.current_scale])

        # 2 - only update binary
        if to_update == 0 or to_update == 2:
            # check if copy exists, if not create ...
            if self.current_scale not in self.view_binary_marked:
                self.view_binary_marked[self.current_scale] = self.view_binary_normal[self.current_scale].copy()

            self.mark_points_in_view(self.view_binary_marked[self.current_scale])

        # 3 - only update thresholds
        if to_update == 0 or to_update == 3:
            # check if copy exists, if not create ...
            if self.current_scale not in self.view_thresholds_marked:
                base_img = self.view_thresholds_normal[self.current_scale]
                self.view_thresholds_marked[self.current_scale] = np.zeros((base_img.shape[0], base_img.shape[1], 3), np.uint8)
                self.view_thresholds_marked[self.current_scale][:, :, 0] = base_img.copy()
                self.view_thresholds_marked[self.current_scale][:, :, 1] = base_img.copy()
                self.view_thresholds_marked[self.current_scale][:, :, 2] = base_img.copy()

            self.mark_points_in_view(self.view_thresholds_marked[self.current_scale])

        # 4 - only update combined
        if to_update == 0 or to_update == 4:
            # check if copy exists, if not create ...
            if self.current_scale not in self.view_combined_marked:
                self.view_combined_marked[self.current_scale] = self.view_combined_normal[self.current_scale].copy()

            self.mark_points_in_view(self.view_combined_marked[self.current_scale])

        # 5 - only update thresholds
        if to_update == 0 or to_update == 5:
            # check if copy exists, if not create ...
            if self.current_scale not in self.view_gray_marked:
                base_img = self.view_gray_normal[self.current_scale]
                self.view_gray_marked[self.current_scale] = np.zeros((base_img.shape[0], base_img.shape[1], 3), np.uint8)
                self.view_gray_marked[self.current_scale][:, :, 0] = base_img.copy()
                self.view_gray_marked[self.current_scale][:, :, 1] = base_img.copy()
                self.view_gray_marked[self.current_scale][:, :, 2] = base_img.copy()

            self.mark_points_in_view(self.view_gray_marked[self.current_scale])

    def update_view_image(self, scaled_changed):
        self.create_normal_view_images(self.current_view)
        self.update_marked_view_images(self.current_view)

        # updates the view controls (image display) with the current scale/
        if self.current_view == 1:
            if self.show_labeled_points:
                view_image = self.view_raw_marked[self.current_scale]
            else:
                view_image = self.view_raw_normal[self.current_scale]

            self.img_raw.set_image(view_image, 0, 0, True, cv2.INTER_NEAREST)
            self.img_raw.visible = True
        else:
            self.img_raw.visible = False

        if self.current_view == 2:
            if self.show_labeled_points:
                view_image = self.view_binary_marked[self.current_scale]
            else:
                view_image = self.view_binary_normal[self.current_scale]

            self.img_binary.set_image(view_image, 0, 0, True, cv2.INTER_NEAREST)
            self.img_binary.visible = True
        else:
            self.img_binary.visible = False

        if self.current_view == 3:
            if self.show_labeled_points:
                view_image = self.view_thresholds_marked[self.current_scale]
            else:
                view_image = self.view_thresholds_normal[self.current_scale]

            self.img_thresholds.set_image(view_image, 0, 0, True, cv2.INTER_NEAREST)
            self.img_thresholds.visible = True
        else:
            self.img_thresholds.visible = False

        if self.current_view == 4:
            if self.show_labeled_points:
                view_image = self.view_combined_marked[self.current_scale]
            else:
                view_image = self.view_combined_normal[self.current_scale]

            self.img_combined.set_image(view_image, 0, 0, True, cv2.INTER_NEAREST)
            self.img_combined.visible = True
        else:
            self.img_combined.visible = False

        if self.current_view == 5:
            if self.show_labeled_points:
                view_image = self.view_gray_marked[self.current_scale]
            else:
                view_image = self.view_gray_normal[self.current_scale]

            self.img_gray.set_image(view_image, 0, 0, True, cv2.INTER_NEAREST)
            self.img_gray.visible = True
        else:
            self.img_gray.visible = False

        # in case that a new scale has been selected
        if scaled_changed:
            self.container_images.recalculate_size()

    def clean_view_cache(self):
        self.view_raw_normal = {}
        self.view_raw_marked = {}
        self.view_gray_normal = {}
        self.view_gray_marked = {}
        self.view_binary_normal = {}
        self.view_binary_marked = {}
        self.view_thresholds_normal = {}
        self.view_thresholds_marked = {}
        self.view_combined_normal = {}
        self.view_combined_marked = {}

    def update_selected_options(self):
        self.container_threshold.visible = False
        if len(self.selected_points) == 0:
            self.btn_points_del.visible = False
        else:
            if len(self.selected_points) == 1:
                # threshold can be set precisely only for one point at a time
                self.container_threshold.visible = True
                threshold_value = self.image_labeled_thresholds[self.selected_points[0]]
                self.threshold_scroll.value = threshold_value
                self.lbl_threshold.set_text("Threshold: " + str(threshold_value))
                self.container_threshold.visible = True

            self.btn_points_del.visible = True

    def set_edition_mode(self, new_edition_mode):
        self.editor_mode = new_edition_mode

        if self.editor_mode == GTBinaryAnnotator.ModeEdition:
            self.container_threshold.visible = (len(self.selected_points) == 1)
            self.container_confirm_buttons.visible = False
            self.container_point_buttons.visible = True
            self.container_state_buttons.visible = True
            self.container_grid_buttons.visible = True

            self.btn_multi_select_clear.visible = False
            self.btn_multi_select_start.visible = True
            self.selected_points = []

        elif self.editor_mode == GTBinaryAnnotator.ModeAddPoint:
            self.container_threshold.visible = False
            self.container_confirm_buttons.visible = True
            self.container_point_buttons.visible = False
            self.container_state_buttons.visible = False
            self.container_grid_buttons.visible = False

            self.lbl_confirm.set_text("Adding New Point")
            self.btn_confirm_accept.visible = False
            self.btn_confirm_cancel.visible = True

            self.btn_multi_select_clear.visible = False
            self.btn_multi_select_start.visible = False

        elif self.editor_mode == GTBinaryAnnotator.ModeConfirmExit or self.editor_mode == GTBinaryAnnotator.ModeConfirmCancel:
            self.container_threshold.visible = False
            self.container_confirm_buttons.visible = True
            self.container_point_buttons.visible = False
            self.container_state_buttons.visible = False
            self.container_grid_buttons.visible = False

            self.lbl_confirm.set_text("Exit without saving?")
            self.btn_confirm_accept.visible = True
            self.btn_confirm_cancel.visible = True

            self.btn_multi_select_clear.visible = False
            self.btn_multi_select_start.visible = False

    def generate_initial_gt_adaptive_otsu(self, image_cut, rows, cols):
        # first, compute the grid that will be used ...
        h, w, _ = image_cut.shape

        tempo_gray = cv2.cvtColor(image_cut, cv2.COLOR_RGB2GRAY)
        partial_gt = np.zeros(image_cut.shape, image_cut.dtype)

        min_row_size = int((h - 1) / rows)
        min_col_size = int((w - 1) / cols)

        neigh_row = min_row_size
        neigh_col = min_col_size

        tempo_mins_gray = np.zeros((rows + 1, cols + 1))
        tempo_maxs_gray = np.zeros((rows + 1, cols + 1))
        tempo_stds_gray = np.zeros((rows + 1, cols + 1))
        tempo_stds_quart = np.zeros((rows + 1, cols + 1))
        tempo_densities = np.zeros((rows + 1, cols + 1))
        tempo_high_locs_adjusted = []
        tempo_low_locs_adjusted = []

        for row in range(rows + 1):
            p_y = row * min_row_size + min(row, (h - 1) % rows)

            for col in range(cols + 1):
                p_x = col * min_col_size + min(col, (w - 1) % cols)

                start_y = max(0, p_y - neigh_row)
                start_x = max(0, p_x - neigh_col)
                end_y = min(h, p_y + neigh_row)
                end_x = min(w, p_x + neigh_col)

                local_patch = tempo_gray[start_y:end_y, start_x:end_x]
                value, binary = cv2.threshold(local_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # deal with background only areas (guess a lower threshold)
                density = 1.0 - ((binary.sum() / 255) / (binary.shape[0] * binary.shape[1]))

                if local_patch.std() >= GTBinaryAnnotator.LocalLowGraySTDev:
                    high_stdev = True
                    max_otsu_density = GTBinaryAnnotator.HighSTDevMaxOtsuDensity
                else:
                    high_stdev = False
                    max_otsu_density = GTBinaryAnnotator.LowSTDevMaxOtsuDensity

                tempo_densities[row, col] = density
                adjusted = False

                while density >= max_otsu_density:
                    adjusted = True
                    value -= 1
                    value2, binary = cv2.threshold(local_patch, value, 255, cv2.THRESH_BINARY)
                    density = 1.0 - ((binary.sum() / 255) / (binary.shape[0] * binary.shape[1]))

                # <TEMPO>
                tempo_mins_gray[row, col] = local_patch.min()
                tempo_maxs_gray[row, col] = local_patch.max()
                tempo_stds_gray[row, col] = local_patch.std()

                if adjusted:
                    if high_stdev:
                        tempo_high_locs_adjusted.append((row, col))
                    else:
                        tempo_low_locs_adjusted.append((row, col))
                # </TEMPO>

                partial_gt[int(p_y), int(p_x), 0] = int(value)
                partial_gt[int(p_y), int(p_x), 1] = 255

        total_points = (rows + 1) * (cols + 1)

        tempo_flat = tempo_stds_gray.flatten()
        for row in range(rows + 1):
            for col in range(cols + 1):
                tempo_stds_quart[row, col] = stats.percentileofscore(tempo_flat, tempo_stds_gray[row, col])

        print("====================================================")
        print("--- gray boundaries --- ")
        print(tempo_mins_gray)
        print(tempo_maxs_gray)
        print("--- std stats --- ")
        print(tempo_stds_gray)
        print(tempo_stds_quart)
        print("--- auto-adjusted --- ")
        print(tempo_densities)
        print("Low: " + str(tempo_low_locs_adjusted))
        print("High: " + str(tempo_high_locs_adjusted))

        return total_points, partial_gt

    def update_point_grid(self, add_undo):
        prev_mask = (self.image_labeled_mask[0].copy(), self.image_labeled_mask[1].copy())
        prev_thresholds = self.image_labeled_thresholds.copy()
        prev_points = self.image_labeled_points.copy()
        prev_K = self.image_K

        # get new grid
        point_count, partial_gt = self.generate_initial_gt_adaptive_otsu(self.base_img_raw, self.point_grid_rows,
                                                                         self.point_grid_cols)

        self.image_labeled_mask = np.nonzero(partial_gt[:, :, 1])
        self.image_labeled_thresholds = partial_gt[self.image_labeled_mask]
        self.image_labeled_thresholds = self.image_labeled_thresholds[:, 0]
        self.image_labeled_points = np.transpose(self.image_labeled_mask)

        self.image_K = 4

        self.grid_changed_update(0, True)

        if add_undo:
            self.undo_stack.append({
                "operation": "grid_applied",
                "old_mask": prev_mask,
                "old_thresholds": prev_thresholds,
                "old_points": prev_points,
                "new_mask": (self.image_labeled_mask[0].copy(), self.image_labeled_mask[1].copy()),
                "new_thresholds": self.image_labeled_thresholds.copy(),
                "new_points": self.image_labeled_points.copy(),
                "old_K": prev_K,
                "new_K": self.image_K,
            })

    def btn_grid_rows_inc_click(self, button):
        # change has no effect on image (yet)
        self.point_grid_rows += 1
        self.lbl_grid_rows.set_text("Rows: " + str(self.point_grid_rows))

    def btn_grid_rows_dec_click(self, button):
        # change has no effect on image (yet)
        if self.point_grid_rows > 1:
            self.point_grid_rows -= 1
            self.lbl_grid_rows.set_text("Rows: " + str(self.point_grid_rows))

    def btn_grid_cols_inc_click(self, button):
        # change has no effect on image (yet)
        self.point_grid_cols += 1
        self.lbl_grid_cols.set_text("Cols: " + str(self.point_grid_cols))

    def btn_grid_cols_dec_click(self, button):
        # change has no effect on image (yet)
        if self.point_grid_cols > 1:
            self.point_grid_cols -= 1
            self.lbl_grid_cols.set_text("Cols: " + str(self.point_grid_cols))

    def btn_grid_update_click(self, button):
        self.update_point_grid(True)

    def apply_offset_all_points(self, delta, add_undo=True):
        prev_threshold_values = self.image_labeled_thresholds.copy()

        n_points = self.image_labeled_thresholds.shape[0]
        for idx in range(n_points):
            if len(self.selected_points) == 0 or idx in self.selected_points:
                # if point is selected, affect selected point
                # if no points are selected, affect the entire grid
                if 0 <= self.image_labeled_thresholds[idx] + delta <= 255:
                    self.image_labeled_thresholds[idx] += delta

        self.grid_changed_update(1, False)

        if add_undo:
            self.undo_stack.append({
                "operation": "offset_applied",
                "old_values": prev_threshold_values,
                "new_values": self.image_labeled_thresholds.copy(),
            })

    def grid_changed_update(self, initial_step, clear_selection):

        # update views
        if clear_selection:
            self.selected_points = []
            self.selection_multi = False
            self.btn_multi_select_clear.visible = False
            self.btn_multi_select_start.visible = True

        self.lbl_points.set_text("Labeled Points: " + str(self.image_labeled_points.shape[0]))
        self.update_base_images(initial_step)
        self.clean_view_cache()

        self.update_view_image(False)
        self.update_selected_options()

    def btn_grid_offset_dec_click(self, button):
        self.apply_offset_all_points(-1)

    def btn_grid_offset_inc_click(self, button):
        self.apply_offset_all_points(1)

    def btn_grid_noise_inc_click(self, button):
        self.point_grid_noise += 1
        self.lbl_grid_noise.set_text("Min Size: " + str(self.point_grid_noise))

        self.update_base_images(3)
        self.clean_view_cache()
        self.update_view_image(False)


    def btn_grid_noise_dec_click(self, button):
        if self.point_grid_noise > 0:
            self.point_grid_noise -= 1
            self.lbl_grid_noise.set_text("Min Size: " + str(self.point_grid_noise))

            self.update_base_images(3)
            self.clean_view_cache()
            self.update_view_image(False)

    def btn_return_cancel_click(self, button):
        if len(self.undo_stack) == 0:
            # return cancel ...
            if self.finished_callback is not None:
                # cancel ...
                self.finished_callback(False, None)
            self.return_screen = self.parent_screen
        else:
            self.set_edition_mode(GTBinaryAnnotator.ModeConfirmCancel)


    def btn_return_accept_click(self, button):
        if self.finished_callback is not None:
            # accept ...
            self.finished_callback(True, self.base_img_binary)

        self.return_screen = self.parent_screen

    def btn_multi_select_start_click(self, button):
        self.selection_multi = True
        self.btn_multi_select_clear.visible = True
        self.btn_multi_select_start.visible = False

    def btn_multi_select_clear_click(self, button):
        self.selection_multi = False
        self.selected_points = []
        self.btn_multi_select_clear.visible = False
        self.btn_multi_select_start.visible = True

        self.update_view_image(False)
        self.update_selected_options()

    def smoothing_scroll_change(self):
        self.smoothing_sigma_space = self.scroll_smoothing_space.value
        self.smoothing_sigma_color = self.scroll_smoothing_color.value

        self.lbl_smoothing_color.set_text("Smoothing Sigma Color: " + str(self.smoothing_sigma_color))
        self.lbl_smoothing_space.set_text("Smoothing Sigma Space: " + str(self.smoothing_sigma_space))

        # update from gray-scale portion
        self.grid_changed_update(-1, False)

    def scroll_smoothing_space_change(self, scroll):
        self.smoothing_scroll_change()

    def scroll_smoothing_color_change(self, scroll):
        self.smoothing_scroll_change()
