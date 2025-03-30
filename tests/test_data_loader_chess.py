import pytest
import torch
from torch.utils.data import DataLoader
from data_loader_chess import train_loader, model, train_dataset, device


@pytest.fixture
def sample_batch():
    """Fetch a single batch from the train_loader."""
    return next(iter(train_loader))


def test_data_loading(sample_batch):
    """Check if data loading works properly."""
    images, labels = sample_batch
    assert images.shape[0] > 0, "DataLoader returned an empty batch!"
    assert len(train_dataset.classes) > 0, "No classes detected in dataset!"


def test_model_fc_layer():
    """Ensure model's FC layer is correctly modified for chess classification."""
    num_classes = len(train_dataset.classes)
    assert model.fc.out_features == num_classes, "FC layer output does not match class count!"


def test_forward_pass(sample_batch):
    """Run a forward pass to check model integrity."""
    images, _ = sample_batch
    images = images.to(device)

    with torch.no_grad():  # Disable gradients for testing
        output = model(images)

    assert output.shape[0] == images.shape[0], "Model output batch size mismatch!"
    assert output.shape[1] == len(train_dataset.classes), "Model output class count mismatch!"

