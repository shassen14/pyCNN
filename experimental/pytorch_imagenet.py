import torch.utils.data
from torchvision import datasets, models, tv_tensors
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import torchvision.transforms as transforms

import torch.nn as nn

all_models = models.list_models()
classification_models = models.list_models(module=models)
print(classification_models)
device: str = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

dataDir = "./data/imagenet/"
dataType = "val"
imgDir = dataDir + dataType

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

inet_train = datasets.ImageNet(dataDir, dataType)

inet_dataset = datasets.ImageFolder(imgDir, transform_train)

idx_to_class = {value: key for key, value in inet_dataset.class_to_idx.items()}

data_loader = torch.utils.data.DataLoader(inet_dataset, batch_size=32, shuffle=True)

model = models.get_model("resnet101").train()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss().to(device)

for i, (imgs, target) in enumerate(data_loader):
    imgs = imgs.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    output = model(imgs)
    loss = criterion(output, target)
    # Put your training logic here
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # print(f"{[img.shape for img in imgs] = }")
    print(i)
    print(output - loss)
    print(f"{loss:.3f}")

    # print(f"{[type(target) for target in targets] = }")
    # for loss_val in loss_dict:
    #     print(f"{loss_val:.3f}")
