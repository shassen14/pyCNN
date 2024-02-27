from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

dataDir = "./data/coco/"
dataType = "val2017"
imgDir = dataDir + dataType
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)
coco_train = datasets.CocoDetection(root=imgDir, annFile=annFile, transform=ToTensor())

img, target = coco_train[0]

print(img)
