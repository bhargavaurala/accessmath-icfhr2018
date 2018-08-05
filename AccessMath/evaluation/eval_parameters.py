

class EvalParameters:
    # Matching Unique Ground Truth CC's in binary key-frame sets
    UniqueCC_global_tran_window = 10 # 10
    UniqueCC_local_trans_window = 3
    UniqueCC_min_translation_fscore = 0.3
    # [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    # [0.50, 0.55, 0.60, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    UniqueCC_min_precision = [0.50, 0.65, 0.80, 0.95]
    UniqueCC_min_recall = [0.50, 0.65, 0.80, 0.95]
    UniqueCC_size_percentiles = [10, 25, 75]
    UniqueCC_min_align_recall = 0.05

    UniqueCC_max_workers = 6

    Report_Summary_Show_Counts = True
    Report_Summary_Show_AVG_per_frame = True
    Report_Summary_Show_Globals = True
    Report_Summary_Show_stats_per_size = False