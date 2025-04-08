import pytest
import torch
import os
from chess_piece_model import ChessPieceModel

# Correct path to the config.properties file
CONFIG_PATH = "./config.properties"

@pytest.fixture(scope="session")
def model_and_loader():
    """Initialize the ChessPieceModel with Google Drive dataset."""
    # Ensure the config file exists
    assert os.path.exists(CONFIG_PATH), f"Config file not found at {CONFIG_PATH}"

    # Initialize ChessPieceModel with the config path
    model = ChessPieceModel(config_path=CONFIG_PATH)
    return model

@pytest.fixture
def sample_batch(model_and_loader):
    """Fetch a single batch from the train_loader."""
    return next(iter(model_and_loader.train_loader))

def test_data_loading(sample_batch, model_and_loader):
    """Check if data loading works properly."""
    images, labels = sample_batch
    assert images.shape[0] > 0, "DataLoader returned an empty batch!"
    assert len(model_and_loader.original_dataset.classes) > 0, "No classes detected in dataset!"

def test_train_val_split(model_and_loader):
    """Ensure that the dataset is split into 80% training and 20% validation."""
    train_size = len(model_and_loader.train_loader.dataset)
    val_size = len(model_and_loader.val_loader.dataset)
    total_size = train_size + val_size

    assert total_size == len(model_and_loader.original_dataset), "Total dataset size mismatch!"
    assert abs(train_size - 0.8 * total_size) < 1, "Training data size is not 80% of total dataset size"
    assert abs(val_size - 0.2 * total_size) < 1, "Validation data size is not 20% of total dataset size"

def test_model_fc_layer(model_and_loader):
    """Ensure model's FC layer is correctly modified for chess classification."""
    num_classes = len(model_and_loader.original_dataset.classes)
    assert model_and_loader.model.fc.out_features == num_classes, "FC layer output does not match class count!"

def test_forward_pass(sample_batch, model_and_loader):
    """Run a forward pass to check model integrity."""
    images, _ = sample_batch
    images = images.to(model_and_loader.device)

    with torch.no_grad():
        output = model_and_loader.model(images)

    assert output.shape[0] == images.shape[0], "Model output batch size mismatch!"
    assert output.shape[1] == len(model_and_loader.original_dataset.classes), "Model output class count mismatch!"


