# script to reorganize AIZZO dataset
'''
-AIZOO                       -AIZOO
    -train                      -train
        a.img                      -images
        a.xml                          a.img
        b.img                          b.img
        b.xml                      -annotations
    -val             --->              a.xml
        c.img                          b.xml
        c.xml                  -val
        d.img                      -images
        d.xml                          c.img
                                       d.img
                                   -annotations
                                       c.xml
                                       d.xml
'''

import os
import shutil

os.chdir("./Data/AIZOO")

subdirs = ["train/images", "train/annotations",
           "val/images", "val/annotations"]
for subdir in subdirs:
    if not os.path.isdir(subdir):
        os.mkdir(subdir)

for item in os.listdir("./train"):
    if item != "images" and item != "annotations":
        if "xml" in item:
            shutil.move("./train/" + item, "./train/annotations/" + item)
        else:
            shutil.move("./train/" + item, "./train/images/" + item)

for item in os.listdir("./val"):
    if item != "images" and item != "annotations":
        if "xml" in item:
            shutil.move("./val/" + item, "./val/annotations/" + item)
        else:
            shutil.move("./val/" + item, "./val/images/" + item)
