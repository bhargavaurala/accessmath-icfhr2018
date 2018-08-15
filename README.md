# Lecture Video Summarization by Extracting Handwritten Content from Whiteboards

Online lecture videos are a valuable resource for students across the world. The ability to find videos based on their content could make them even more useful. Methods for automatic extraction of this content reduce the amount of manual effort required to make indexing and retrieval of such videos possible. We adapt a deep learning based method for scene text detection, for the purpose of detection of handwritten text, math expressions and sketches in lecture videos.

![Our methodology](Methodology.png)

More details can be found in our [paper](https://buffalo.box.com/s/nhjjwpj1j4tlvwc762a65tsimmnyn7d2).

## To reproduce the results in the paper (Table 2):

- Download AccessMath [Dataset](https://buffalo.box.com/s/usa30o2o0oojcslfrkxfyogvbqdedktj) and copy into project root.

- Download our Handwritten Content Detector [model](https://buffalo.box.com/s/elz4wj1favsa24apcjiz0co7ash5qry6) and structure [file](https://buffalo.box.com/s/cb0m7nr1dcmyt9610642yvzncf3ectqd) and place in `models/text_detection`.

- Setup [AccessMath-TextBoxes](https://github.com/bhargavaurala/accessmath-textboxes). If needed, generate training LMDBs.

- Run the following scripts:

-- Export video into still frames for generating training samples for text detector by running 
```
python pre_ST3D_v2.0_00_export_frames.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run Text Detection on exported still testing video frames
```
python pre_ST3D_v2.0_01_text_detection.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run stability analysis on detected handwritten regions
```
python pre_ST3D_v2.0_02_td_stability.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run coarse-grained temporal analysis
```
python pre_ST3D_v2.0_03_td_bbox_grouping.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run binarization with reconstruction (bringing back occluded content - part 2 of Table 2 - recommended)
```
python pre_ST3D_v2.0_04_td_ref_binarize.py test_data/databases/db_AccessMath2015.xml -d testing
```
   OR (without reconstruction part 1 of Table 2)
```
python pre_ST3D_v2.0_04_td_raw_binarize.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run fine-grained temporal refinement
```
python pre_ST3D_v2.0_05_cc_analysis.py test_data/databases/db_AccessMath2015.xml -d testing
python pre_ST3D_v2.0_06_cc_grouping.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Run conflict minimization
```
python pre_ST3D_v2.0_07_vid_segmentation.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- Generate final keyframe summaries and evaluation results
```
python pre_ST3D_v2.0_08_generate_summary.py test_data/databases/db_AccessMath2015.xml -d testing
```
-- The final summary keyframes can be found in `output/`

## To retrain on custom data:

- Download the SSD [model](https://buffalo.box.com/s/3rklcvppkuw63jkp4538nbf6u6gj9zqm) for VOC object class detection and place in `models/person_detection`

- Clone the [SSD PyTorch](https://github.com/amdegroot/ssd.pytorch) repository and set it up. Add this directory to `$PYTHONPATH`
```
export PYTHONPATH=/path/to/ssd.pytorch/:$PYTHONPATH
```

- Run the following scripts:

-- Export video into still frames 
```
python pre_ST3D_v2.0_00_export_frames.py test_data/databases/db_AccessMath2015.xml -d "training, testing"
```
-- Generate person detection bounding boxes on training set and add to annotations

```
python gt_PD_01_detect_speaker.py test_data/databases/db_AccessMath2015.xml -d training
python gt_PD_02_add_speaker_to_annotations.py test_data/databases/db_AccessMath2015.xml -d training
```
-- Generate ground truth annotations by removing text region annotations that are occluded by speaker
```
python pre_ST3D_v2.0_00_export_frames_annotations.py test_data/databases/db_AccessMath2015.xml -d training
```
-- Generate trained model using the procedure described in [AccessMath-TextBoxes](https://github.com/bhargavaurala/accessmath-textboxes/blob/master/README.md)

-- Follow procedure to reproduce Table 2 starting with `01_text_detection.py` 

