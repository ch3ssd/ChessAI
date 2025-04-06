import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import requests
import zipfile
from io import BytesIO


class ChessPieceModel:
    def __init__(self, drive_url: str = None, data_dir: str = "./TrainingImagesPreprocessed", batch_size: int = 32,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size

        # Ensure dataset is available (this step is handled by CI)
        if drive_url:
            self.download_and_extract_data(drive_url)

        self.train_loader, self.val_loader = self.initialize_data_loader()
        self.model = self.initialize_model()

    def download_and_extract_data(self, drive_url):
        """Download and extract dataset if not available."""
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            print("Downloading dataset from Google Drive...")
            response = requests.get(drive_url, stream=True)
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir.parent)  # Extract to the parent of data_dir
                print("Dataset downloaded and extracted.")
            else:
                raise RuntimeError("Failed to download dataset from Google Drive.")

    def get_data_transforms(self):
        """Returns data augmentation and normalization transformations for training and validation."""
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transforms, val_transforms

    def initialize_data_loader(self):
        """Initialize the data loader with transformations and dataset."""
        train_transforms, val_transforms = self.get_data_transforms()

        # Load the entire dataset
        full_dataset = datasets.ImageFolder(root=str(self.data_dir), transform=train_transforms)

        # Split the dataset into 80% training and 20% validation
        train_size = int(0.8 * len(full_dataset))  # 80% for training
        val_size = len(full_dataset) - train_size  # 20% for validation
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply validation transformations to the validation dataset
        val_dataset.dataset.transform = val_transforms

        # Create DataLoaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader

    def initialize_model(self):
        """Load a pre-trained ResNet-50 model and modify the fully connected layer for chess classification."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features

        num_classes = len(self.train_loader.dataset.dataset.classes)

        model.fc = nn.Linear(num_features, num_classes)
        return model.to(self.device)

    def load_data_into_model(self):
        """Load a batch of augmented data and pass it through the model."""
        sample_images, sample_labels = next(iter(self.train_loader))  # Get a batch from training data
        sample_images = sample_images.to(self.device)

        with torch.no_grad():
            output = self.model(sample_images)

        print(f"Loaded batch shape: {sample_images.shape}")
        print(f"Number of classes detected: {len(self.train_loader.dataset.classes)}")
        print(f"Model output shape: {output.shape}")


