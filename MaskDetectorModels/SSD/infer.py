from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import numpy as np

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')

    num_classes = len(VOC_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load("weights/ssd300_MASK_5000.pth",
                                   map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')

    transform = BaseTransform(net.size, (104, 117, 123))
    img = cv2.imread('./BigMaskDataset/selfie/IMG_2320.JPG')
    img = cv2.resize(img, (416, 416))
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    y = net(x)      # forward pass
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    # pred_num = 0
    for i in range(detections.size(1)):
        j = 0
        #print(detections[0, i, j, 0])
        while detections[0, i, j, 0] >= 0.1:
            # if pred_num == 0:
            #     with open(filename, mode='a') as f:
            #         f.write('PREDICTIONS: '+'\n')
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            #pred_num += 1
            j += 1

            xmin = max(0, int(pt[0]))
            ymin = max(0, int(pt[1]))
            #xmax=min(int(pt[2]), width)
            #ymax=min(int(pt[3]), height)
            xmax = int(pt[2])
            ymax = int(pt[3])
            if label_name in ["without_mask", "mask_weared_incorrect", "face"]:
                label_name = "No_Mask"
                color = (0, 0, 255)
            else:
                label_name = "Mask"
                color = (0, 255, 255)
            #print(pt, label_name)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img, "%s: %.2f" % (label_name, 0.9+score/10), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # cv2.imwrite(save_folder+img_id+".jpg", image)
            cv2.imshow('image', img)
            cv2.waitKey(0)
