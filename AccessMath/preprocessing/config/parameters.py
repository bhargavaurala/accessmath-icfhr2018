
#============================================================================
# Global file with fixed parameter values for AccessMath
#
# Kenny Davila
# - Created:  October 9, 2015
# - Modified:
#       - March 2017
#       - June 2017
#
#============================================================================

import numpy as np

class Parameters:
    # video sampling
    Sampling_FPS = 1.0 # 1.0

    # binarization
    Bin_method = 4 # New background subtraction

    Bin_dark_background = False
    Bin_sigma_color = 4.0
    Bin_sigma_space = 4.0
    Bin_bluring_ksize = 3 # 3
    Bin_bgsub_threshold = 0.89 # 0.89
    Bin_min_pixels = 4 # 6

    Bin_disk_size = 14 # 14
    Bin_chalk_threshold = 20

    # CC stability and CC groups
    CCStab_stability_coefficient = 0.9
    CCStab_stability_min_recall = 0.925
    CCStab_stability_min_precision = 0.925
    CCStab_min_times = 3
    CCStab_max_gap = 85 # 85
    CCGroup_temporal_window = 5
    CCGroup_min_image_threshold = 0.5

    # Background Removal
    # (based method using CC's and thresholds)
    BGRem_high_threshold = 0.80 # 0.80
    BGRem_low_threshold = 0.40  # 0.40
    BGRem_min_bg_cc_size = 200
    BGRem_close_radius = 5
    # (Refinement using Nearest Neighbors)
    BGRem_KNN_content_prc = 0.40  # 40% of the video
    # BGRem_KNN_noise_time = 30  # 30 seconds
    BGRem_KNN_noise_prc = 0.01  # % of the video
    BGRem_KNN_neighbors = 100
    BGRem_KNN_dilation = 10
    BGRem_KNN_min_region_size = 0.10

    # Video Segmentation
    VSeg_method = 2 # 1 - Sums, 2 - Conflicts
    VSeg_Sum_min_segment = 10
    VSeg_Sum_min_erase_ratio = 0.05 # 0.15
    VSeg_Conf_min_conflicts = 3    # minimum conflicst to accept split
    VSeg_Conf_min_split = 50       # minimum segment length to consider splitting it
    VSeg_Conf_min_len = 25       # minimum segment length to accept split. (related to VSeg_Conf_min_split)
    VSeg_Conf_weights = 0          # 0 - simple count, 1 - matched pixels, 2 - unmatched pixels
    VSeg_Conf_weights_time = False  # multiply weights by the size of the gap
    # Keyframe extraction
    KFExt_min_length = 5

    # Shot boundary detection
    SBDet_method = 1
    SBDet_features = 2
    SBDet_distance = 2
    SBDet_lum_features = 1024
    SBDet_lum_threshold = 1500
    SBDet_color_channel_bins = 4
    SBDet_color_channel_threshold = 0.175
    SBDet_color_combined_bins = 4
    SBDet_color_combined_threshold = 0.01

    # Log Motion Difference
    LMot_min_percentile = 5 # 25

    # ROI Detection
    ROIDet_fps = 0.1
    ROIDet_temporal_blur_K = 11
    ROIDet_bin_threshold = 5 # 8
    ROIDet_edge_min_threshold = 30
    ROIDet_edge_max_threshold = 50
    ROIDet_Hough_rho = 1
    ROIDet_Hough_theta = np.pi / 180
    ROIDet_Hough_min_intersections = 50
    ROIDet_Hough_min_line_length = 100
    ROIDet_Hough_max_line_gap = 10
    ROIDet_Hough_diag_threshold = (15.0 / 180.0) * np.pi
    ROIDet_workers = 6

    # Background removal using detected ROI
    ROIBGRem_min_overlap = 1.0

    # model files
    Model_PersonDetection = 'models/person_detection/'
    Model_TextDetection = 'models/text_detection/'

    # outputs ...
    Output_Binarize = "tempo_binary_"
    Output_GTBinarize = "tempo_gt_binary_"
    Output_CCStability = "tempo_stability_"
    Output_ST3D = "tempo_cc_ST3D_"
    Output_CCConflicts = "tempo_cc_conflicts_"
    Output_CCReconstructed = "tempo_bin_reconstructed_"
    Output_BG_Removal = "tempo_no_bg_binary_"
    Output_GT_BG_Removal = "tempo_gt_no_bg_binary_"
    Output_Vid_Segment = "tempo_intervals_"
    Output_SUM_Vid_Segment = "tempo_sum_intervals_"
    Output_Ext_Keyframes = "tempo_segments_"
    Output_Log_Motion = "log_motion_"
    Output_SBDetection = "shot_boundaries_"

    Output_ROI_Detection = "ROI_mask_"

    Output_FrameExport = "AccessMathVOC/" # make sure to add a trailing backslash if you change this
    Output_PersonDetection = "person_detection_"
    Output_TextDetection = "text_detection_"
    Output_TDStability = "td_bboxes_stability_"
    Output_TDRefined = "td_refined_"

    MLBin_sampling_mode = 2
    MLBin_sampling_patches_per_frame = 20000
    MLBin_sampling_fg_proportion = 0.5
    MLBin_sampling_bg_close_prop = 0.9
    MLBin_sampling_bg_board_prop = 1.0

    MLBin_train_workers = 7

    MLBin_patch_size = 7    
    MLBin_rf_n_trees = 16
    MLBin_rf_max_depth = 12
    MLBin_rf_max_features = 32    
    MLBin_classifier_file = "output/classifier/RF_T16_D12_F32_w7x7.dat"
        
    MLBin_sigma_color = 13.5
    MLBin_sigma_space = 4.0
    MLBin_median_blur_k = 33
    MLBin_dark_background = False
    MLBin_hysteresis = True

    TDStab_min_comb_box_ratio = 0.5
    TDStab_min_temp_box_IOU = 0.5        # minimum spatial IOU to combine boxes across time
    TDStab_max_temporal_gap = 85
    TDStab_min_confidence = 0.65 # 0.65
    TDStab_min_times = 3
    TDGroup_temporal_window = 5
    TDBin_ML_binarization = False

