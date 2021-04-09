# cs1430-final-project

## Nose+Eyes+FaceDetector

A collection of HAAR cascade classifiers and scripts to run them.


## NoseDetector+MaskDetector

A simple but working script for ensembling a HAAR-based nose detector + a CNN-based mask detector

The pretrained CNN comes from [AIZOOTech/FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection). It's super light weight (only 1M params). This is a great architecture we can refer to.
The local copy of the model can be found [here](NoseDetector+MaskDetector/MainModel.py).

## SSDMaskDetector

eval_backbone.py is a working version of this [tutorial](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/). This is more heavyweight, and would likely need to be served, rather than deployed in the client. I think we could tweak this [model source file](SSDMaskDetector/SSD300.py) and retrain on our datasets. 

## TrainingPipeline

A basic training pipeline for our mask detector. This was made by adding new tasks (e.g. mobileNet) to code from Proj4.
