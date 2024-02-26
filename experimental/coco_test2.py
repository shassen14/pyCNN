import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as PImage
import cv2

"""
Following COCO tutorial, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
"""

dataDir = "./data/coco/"
dataType = "val2017"
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# get all images
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds)

for info in img:
    # obtain hardcoded file path
    image_path = "./data/coco/" + dataType + "/" + info["file_name"]

    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, (640, 480))
    # image_concat = np.concatenate((image, image_resize), axis)

    # Window name in which image is displayed
    window_name = "image"

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, image_resize)
    # cv2.imshow("bob", image)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    else:
        continue
