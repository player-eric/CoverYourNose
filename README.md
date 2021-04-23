# Cover Your Nose


A group project for [CSCI 1430 Introduction to Computer Vision, Spring 2021](https://browncsci1430.github.io/webpage/index.html)

<p align="center"><img src="/AppOverview/AppOverview1.jpeg" align="center" width="512" height="384" align="center" /></p>


## Introduction

Despite the significant vaccination process, people will still need to wear
masks for a long time when they are outdoors. Yet currently many people are not wearing masks in proper ways, and most of them are unconscious of this. So in this project, a web app with CV models running on backend is developed to help people check if they are wearing masks correctly.

Visit the app at [coveryournose.xyz](http://coveryournose.xyz/).

## Code structure

### Dataset

Scripts for downloading the big mask / no mask dataset we created.

### HAAR-Nose+Eyes+FaceDetectors

A collection of HAAR cascade classifiers and scripts to run them.

### MaskDetectorModels

Three different models mask detection: [Faster R-CNN](https://arxiv.org/abs/1506.01497), [SSD](https://arxiv.org/abs/1512.02325), and [Mobilenet](https://arxiv.org/abs/1704.04861).

### Nose+MaskDetector

Ensemble of a HAAR-based nose detector an a pretrained SSD model for mask detection.

A local copy of the architecture of this model can be found [here](NoseDetector+MaskDetector/MainModel.py).

### FlaskServers

A Flask server application with Nose+MaskDetector running on the backend, ready to deploy to cloud with Docker. Together with a tensorflow MNIST classifer server for rapid prototyping.

## References

[AIZOOTech/FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection)

[Pytorch - FasterRCNN](https://www.kaggle.com/daniel601/pytorch-fasterrcnn)

[amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
