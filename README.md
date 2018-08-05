# Lecture Video Summarization by Extracting Handwritten Content from Whiteboards

Online lecture videos are a valuable resource for students across the world. The ability to find videos based on their content could make them even more useful. Methods for automatic extraction of this content reduce the amount of manual effort required to make indexing and retrieval of such videos possible. We adapt a deep learning based method for scene text detection, for the purpose of detection of handwritten text, math expressions and sketches in lecture videos.

[Our methodology](Methodology.png)

More details can be found in our [paper]().

## To reproduce the results in the paper:

- Download AccessMath Dataset
[Link]()

- Download our Handwritten Content Detector [model]() and structure [file]() and place in `models/text_detection`.

- Run the following scripts:

-- 

-- 

## To retrain on custom data:

- Download the SSD [model]() for VOC object class detection and place in `models/person_detection`.

- Run the following scripts:

-- 

-- 

