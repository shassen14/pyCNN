import torch.utils.data
from torchvision import datasets, models, tv_tensors
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

dataDir = "./data/coco/"
dataType = "val2017"
imgDir = dataDir + dataType
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

coco_train = datasets.CocoDetection(root=imgDir, annFile=annFile, transforms=transforms)

coco_dataset = datasets.wrap_dataset_for_transforms_v2(
    coco_train, target_keys=("boxes", "labels", "masks")
)

data_loader = torch.utils.data.DataLoader(
    coco_dataset,
    batch_size=2,
    # We need a custom collation function here, since the object detection
    # models expect a sequence of images and target dictionaries. The default
    # collation function tries to torch.stack() the individual elements,
    # which fails in general for object detection, because the number of bouding
    # boxes varies between the images of a same batch.
    collate_fn=lambda batch: tuple(zip(*batch)),
)

model = models.get_model(
    "maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None
).train()

for imgs, targets in data_loader:
    loss_dict = model(imgs, targets)
    # Put your training logic here

    print(f"{[img.shape for img in imgs] = }")
    print(f"{[type(target) for target in targets] = }")
    for name, loss_val in loss_dict.items():
        print(f"{name:<20}{loss_val:.3f}")
