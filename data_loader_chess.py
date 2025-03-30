import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

class ChessPieceModel:
    def __init__(self, data_dir: str, batch_size: int = 32, device: torch.device = None):
        # Set device (GPU if available, otherwise CPU)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths to dataset
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size

        # Initialize the model and data loader
        self.train_loader = self._initialize_data_loader()
        self.model = self._initialize_model()

    def _initialize_data_loader(self):
        """Initialize the data loader with transformations and dataset."""
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
        train_dataset = datasets.ImageFolder(root=str(self.data_dir), transform=train_transforms)

        # Create data loader
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def _initialize_model(self):
        """Load a pre-trained ResNet-50 model and modify the fully connected layer for chess classification."""
        # Load pre-trained ResNet-50 model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features

        # Modify the fully connected layer for chess classification
        num_classes = len(self.train_loader.dataset.classes)  # Auto-detect number of classes
        model.fc = nn.Linear(num_features, num_classes)

        # Move model to device (GPU if available)
        return model.to(self.device)

    def load_data_into_model(self):
        """Load a batch of augmented data and pass it through the model."""
        # Get a sample batch of data
        sample_images, sample_labels = next(iter(self.train_loader))
        sample_images = sample_images.to(self.device)

        # Pass the images through the model (forward pass)
        with torch.no_grad():  # No need to calculate gradients
            output = self.model(sample_images)

        print(f"Loaded batch shape: {sample_images.shape}")
        print(f"Number of classes detected: {len(self.train_loader.dataset.classes)}")
        print(f"Model output shape: {output.shape}")


