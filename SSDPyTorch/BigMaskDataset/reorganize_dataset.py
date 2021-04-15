# script for reorganizing dataset
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

subdirs = ["train/JPEGImages", "train/Annotations",
           "val/JPEGImages", "val/Annotations"]
for subdir in subdirs:
    if not os.path.isdir(subdir):
        os.mkdir(subdir)


for (idx, item) in enumerate(list(os.listdir("./train"))):
    if item != "JPEGImages" and item != "Annotations" and item != "ImageSets":
        if "xml" in item:
            shutil.move("./train/" + item,
                        "./train/Annotations/" + item)
        else:
            shutil.move("./train/" + item,
                        "./train/JPEGImages/" + item)
# print(fns)
fns = []
for item in os.listdir("./train/Annotations"):
    fns.append(item[:-4] + "\n")

with open("./train/ImageSets/Main/train.txt", "w") as outf:
    outf.writelines(fns)

for (idx, item) in enumerate(list(os.listdir("./val"))):
    if item != "JPEGImages" and item != "Annotations" and item != "ImageSets":
        if "xml" in item:
            shutil.move("./val/" + item, "./val/Annotations/" +
                        item)
        else:
            shutil.move("./val/" + item, "./val/JPEGImages/" +
                        item)

fns = []
for item in os.listdir("./val/Annotations"):
    fns.append(item[:-4] + "\n")

with open("./val/ImageSets/Main/val.txt", "w") as outf:
    outf.writelines(fns)
