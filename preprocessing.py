import numpy as np
import cv2
import os
import csv
from image_processing import func

# Create necessary directories if they don't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

path = "train"
path1 = "data"
a = ['label']

# Create pixel header names
for i in range(64*64):
    a.append("pixel" + str(i))

label = 0
var = 0
c1 = 0
c2 = 0

# Walk through directories
for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for (direcpath, direcnames, files) in os.walk(path + "/" + dirname):
            # Create necessary directories for train and test
            if not os.path.exists(path1 + "/train/" + dirname):
                os.makedirs(path1 + "/train/" + dirname)
            if not os.path.exists(path1 + "/test/" + dirname):
                os.makedirs(path1 + "/test/" + dirname)

            # Split into training and testing sets
            num = 100000000000000000
            i = 0
            for file in files:
                var += 1
                actual_path = path + "/" + dirname + "/" + file
                actual_path1 = path1 + "/" + "train/" + dirname + "/" + file
                actual_path2 = path1 + "/" + "test/" + dirname + "/" + file
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)

                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image)
                    
                i += 1
        label += 1

print(var)
print(c1)
print(c2)