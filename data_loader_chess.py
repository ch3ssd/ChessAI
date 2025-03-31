import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import requests
import zipfile
from io import BytesIO


class ChessPieceModel:
    def __init__(self, drive_url: str, data_dir: str = "./TrainingImagesPreprocessed", batch_size: int = 32,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size

        # Ensure dataset is available
        self._download_and_extract_data(drive_url)

        self.train_loader = self._initialize_data_loader()
        self.model = self._initialize_model()

    def _download_and_extract_data(self, drive_url):
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

    def _initialize_data_loader(self):
        """Initialize the data loader with transformations and dataset."""
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(root=str(self.data_dir), transform=train_transforms)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def _initialize_model(self):
        """Load a pre-trained ResNet-50 model and modify the fully connected layer for chess classification."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        num_classes = len(self.train_loader.dataset.classes)
        model.fc = nn.Linear(num_features, num_classes)
        return model.to(self.device)

    def load_data_into_model(self):
        """Load a batch of augmented data and pass it through the model."""
        sample_images, sample_labels = next(iter(self.train_loader))
        sample_images = sample_images.to(self.device)

        with torch.no_grad():
            output = self.model(sample_images)

        print(f"Loaded batch shape: {sample_images.shape}")
        print(f"Number of classes detected: {len(self.train_loader.dataset.classes)}")
        print(f"Model output shape: {output.shape}")


