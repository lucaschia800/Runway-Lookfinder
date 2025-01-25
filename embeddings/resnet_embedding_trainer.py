"""
ResNet-50 Fine-tuning for Multi-Label Classification

This script fine-tunes a ResNet-50 model on the DeepFashion dataset for multi-label
classification tasks using PyTorch with mixed precision training and early stopping.
"""

import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import copy

# Configure CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FashionDataset(Dataset):
    """DeepFashion dataset loader for multi-label classification.
    
    Args:
        img_paths (list): List of image file paths
        anno_list (list): List of annotation strings
        transform (callable, optional): Optional transform to apply to images
    """
    
    def __init__(self, img_paths: list, anno_list: list, transform: transforms.Compose = None):
        self.transform = transform
        self.image_paths = img_paths
        self.annotations = anno_list

    def __getitem__(self, index: int) -> tuple:
        """Load and process single image-label pair."""
        # Load image
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Process labels
        label_str = self.annotations[index].strip()
        labels = [int(lbl) for lbl in label_str.split()]
        return image, torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_paths)


def create_transforms(train: bool = True) -> transforms.Compose:
    """Create image transformations pipeline.
    
    Args:
        train (bool): Whether to create training transforms
    
    Returns:
        transforms.Compose: Composition of image transformations
    """
    base_transforms = [
        transforms.ToImage(),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        base_transforms.insert(3, transforms.RandomHorizontalFlip(0.5))
        base_transforms.insert(4, transforms.RandomVerticalFlip(0.5))
        
    return transforms.Compose(base_transforms)


def initialize_model(num_classes: int = 26) -> nn.Module:
    """Initialize ResNet-50 model with custom classifier.
    
    Args:
        num_classes (int): Number of output classes
    
    Returns:
        nn.Module: Configured ResNet-50 model
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_data_paths(file_path: str) -> list:
    """Load file paths from text file.
    
    Args:
        file_path (str): Path to text file containing paths
    
    Returns:
        list: List of stripped lines from file
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, scaler: GradScaler, device: torch.device) -> float:
    """Execute single training epoch.
    
    Returns:
        float: Average epoch loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(loader):
        # Clear GPU cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
        # Move data to device
        images, targets = images.to(device), targets.to(device)
        
        # Mixed precision training
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate model on validation set.
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, targets).item()
            
    return total_loss / len(loader)


def main():
    """Main training workflow."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Load data paths
    train_images = load_data_paths("deepfashion_annos/train.txt")
    train_annos = load_data_paths("deepfashion_annos/train_attr.txt")
    val_images = load_data_paths("deepfashion_annos/val.txt")
    val_annos = load_data_paths("deepfashion_annos/val_attr.txt")

    # Create datasets and loaders
    train_dataset = FashionDataset(train_images, train_annos, create_transforms(train=True))
    val_dataset = FashionDataset(val_images, val_annos, create_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=6)

    # Initialize model
    model = initialize_model()
    model.load_state_dict(torch.load('resnet_50.pth'))
    
    # Unfreeze specific layers
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name:
            param.requires_grad = True

    # Configure optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': 5e-6},
        {'params': model.layer3.parameters(), 'lr': 5e-6},
        {'params': model.fc.parameters(), 'lr': 5e-5}
    ], weight_decay=1e-1)
    
    # Training loop
    best_weights = None
    best_loss = float('inf')
    patience = 10
    scaler = GradScaler()
    
    for epoch in range(200):
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = 10
        else:
            patience -= 1
            
        if patience <= 0:
            print("Early stopping triggered")
            break

    # Save final model
    torch.save(best_weights or model.state_dict(), "resnet_50_finetune.pth")


if __name__ == "__main__":
    main()