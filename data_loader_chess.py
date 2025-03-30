import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from pathlib import Path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to dataset
DATA_DIR = Path("./TrainingImagesPreprocessed").resolve()

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet input
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformations
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# Load dataset
train_dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Load pre-trained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features

# Modify the fully connected layer for chess classification
num_classes = len(train_dataset.classes)  # Auto-detect number of classes
model.fc = nn.Linear(num_features, num_classes)

# Move model to device (GPU if available)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if everything is working
sample_images, sample_labels = next(iter(train_loader))
print(f"Loaded batch shape: {sample_images.shape}")
print(f"Number of classes detected: {num_classes}")
