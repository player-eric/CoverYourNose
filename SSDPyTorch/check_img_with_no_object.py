# https: // github.com/amdegroot/ssd.pytorch/issues/224  # issuecomment-416511544
import argparse
import sys
import cv2
import os

import os.path as osp
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--root', help='Dataset root directory path')

args = parser.parse_args()

CLASSES = (  # always index 0
    'face', 'face_mask', "with_mask", "without_mask", "mask_weared_incorrect")

annopath = osp.join('%s', 'Annotations', '%s.{}'.format("xml"))
imgpath = osp.join('%s', 'JPEGImages',  '%s.{}'.format("jpg"))


def vocChecker(image_id, width, height, keep_difficult=False):

    global to_delete

    target = ET.parse(annopath % image_id).getroot()
    res = []

    for obj in target.iter('object'):

        difficult = int(obj.find('difficult').text) == 1

        if not keep_difficult and difficult:
            continue

        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []

        for i, pt in enumerate(pts):

            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            cur_pt = float(cur_pt) / \
                width if i % 2 == 0 else float(cur_pt) / height

            bndbox.append(cur_pt)

        # print(name)
        label_idx = dict(zip(CLASSES, range(len(CLASSES))))[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
    # print(res)
    try:
        #print(np.array(res)[:, 4])
        a = np.array(res)[:, 4]
        #print(np.array(res)[:, :4])
        b = np.array(res)[:, :4]
    except IndexError:
        print("INDEX ERROR HERE !\n")
        print(image_id)
        to_delete.append(image_id)
        # exit(0)
    return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


if __name__ == '__main__':
    to_delete = []
    i = 0
    for name in sorted(os.listdir(osp.join(args.root, 'Annotations'))):
        # as we have only one annotations file per image
        i += 1
        # print(name)
        #print(imgpath % (args.root, ".".join(name.split('.')[:-1])))
        img = cv2.imread(imgpath % (args.root, ".".join(name.split('.')[:-1])))
        height, width, channels = img.shape
        # print("path : {}".format(annopath %
        #                         (args.root, ".".join(name.split('.')[:-1]))))
        res = vocChecker((args.root, ".".join(
            name.split('.')[:-1])), height, width)
    print("Total of annotations : {}".format(i))
    print(to_delete)
