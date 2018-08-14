# Lecture Video Summarization by Extracting Handwritten Content from Whiteboards

Online lecture videos are a valuable resource for students across the world. The ability to find videos based on their content could make them even more useful. Methods for automatic extraction of this content reduce the amount of manual effort required to make indexing and retrieval of such videos possible. We adapt a deep learning based method for scene text detection, for the purpose of detection of handwritten text, math expressions and sketches in lecture videos.

![Our methodology](Methodology.png)

More details can be found in our [paper](https://buffalo.box.com/s/nhjjwpj1j4tlvwc762a65tsimmnyn7d2).

## To reproduce the results in the paper:

- Download AccessMath [Dataset](https://buffalo.box.com/s/usa30o2o0oojcslfrkxfyogvbqdedktj) and copy into project root.

- Download our Handwritten Content Detector [model](https://buffalo.box.com/s/elz4wj1favsa24apcjiz0co7ash5qry6) and structure [file]() and place in `models/text_detection`.

- Run the following scripts:

-- 

-- 

## To retrain on custom data:

- Download the SSD [model]() for VOC object class detection and place in `models/person_detection`.

- Run the following scripts:

-- Export video into still frames for generating training samples for text detector by running 
```
python pre_ST3D_v2.0_00_export_frames.py test_data/databases/db_AccessMath2015.xml
```
-- 

