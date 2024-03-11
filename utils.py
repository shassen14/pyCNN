import os
import torch


###############################################################################
def get_folder_path(folder: str) -> str:
    """Obtain folder path. Side effect is creating one if it doesn't exist"""
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


###############################################################################
def get_file_path(folder: str, file: str) -> str:
    """Obtain file path. Side effect is creating directory if it doesn't exist"""
    file_path = os.path.join(get_folder_path(folder), file)
    return file_path


###############################################################################
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == labels)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}%".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    100 * accuracy / len(inputs),
                )
            )


###############################################################################
def validate(model, val_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset) / val_loader.batch_size
    accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(val_loader.dataset), accuracy
        )
    )
