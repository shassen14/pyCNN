import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as PImage
import cv2

"""
Following COCO tutorial, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
"""

dataDir = "./data/coco/"
dataType = "train2017"
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat["name"] for cat in cats]
print("COCO categories: \n{}\n".format(" ".join(nms)))

nms = set([cat["supercategory"] for cat in cats])
print("COCO supercategories: \n{}".format(" ".join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=["person", "dog", "skateboard"])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
print(img)

# obtain hardcoded file path
image_path = "./data/coco/" + dataType + "/" + img["file_name"]

image = cv2.imread(image_path)

# Window name in which image is displayed
window_name = "image"

# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
