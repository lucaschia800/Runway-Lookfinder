import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import h5py
import io
import numpy as np
import matplotlib.pyplot as plt
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import Subset
import copy

class CustomDataset(Dataset):
    def __init__(self, hdf5_file, transforms=None):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.transforms = transforms
        self.images = self.hdf5_file['images']
        self.bounding_boxes = self.hdf5_file['bounding_boxes']
        self.labels = self.hdf5_file['labels']

    def __getitem__(self, idx):

        image = self.images[idx]
        image = Image.open(io.BytesIO(image)).convert("RGB")

        boxes = np.array(self.bounding_boxes[idx], dtype=np.float32).reshape(-1, 4)
        labels = [int(label) for label in self.labels[idx].decode('utf-8').split(",")]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)


def get_transform(train):
    transforms = []
    # Convert PIL image to PyTorch tensor
    transforms.append(lambda x, target: (F.to_tensor(x), target))
    if train:
        transforms.append(lambda x, target: (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(x), target))
        transforms.append(lambda x, target: (T.Normalize( mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])(x), target)) #I believe this needs to be fixed


    return Compose(transforms)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one for transfer learning
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = False

    return model

def train_model(model, data_loader, val_loader, optimizer, device, batch_size, num_epochs=100):
    model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    patience = 10
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(data_loader):

            if i % 100 == 0:
              print('Photos Processed:' + str(i * batch_size))

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")
        if epoch > 10:
          val_loss = validate(model, val_loader, device)
          compute_map(model, val_loader, device)

          if val_loss < best_loss:
              best_loss = val_loss
              best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
              patience = 10  # Reset patience counter
          else:
              patience -= 1
              if patience == 0:
                  print('Early stop at epochL:' + epoch)
                  early_stop = True
                  break

    if early_stop:
        return best_model_weights
    else:
        return model.state_dict()

def validate(model, data_loader, device):
    model.train()
    val_loss = 0
    print('Validating')

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    avg_val_loss = val_loss / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def compute_map(model, data_loader, device):
    print('Calculating MAP')
    metric = MeanAveragePrecision(iou_type = 'bbox')
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            predictions = [{k: v.to(device) for k, v in prediction.items()} for prediction in predictions]

            metric.update(predictions, targets)
        print(metric.compute())

# Main execution
if __name__ == "__main__":
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Create datasets and instantiate dataloader
    batch_size = 8
    training_data = CustomDataset(hdf5_file = '/content/h5 files/train.h5', transforms=get_transform(train=True))
    subset_data = torch.utils.data.Subset(training_data, range(10000))
    train_loader = DataLoader(subset_data, batch_size= batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers= 4 )

    validation_data = CustomDataset(hdf5_file = '/content/h5 files/validation.h5' , transforms=get_transform(train=False))
    subset_val = torch.utils.data.Subset(validation_data, range(2000))
    val_loader = DataLoader(subset_val, batch_size = 8, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 14
    # Get model
    model = get_model(num_classes)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


    torch.save(train_model(model, train_loader, val_loader, optimizer, device, batch_size),'rcnn_resnet_model.pth')
