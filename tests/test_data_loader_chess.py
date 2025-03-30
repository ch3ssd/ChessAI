import pytest
import torch
from data_loader_chess import ChessPieceModel


@pytest.fixture
def model_and_loader():
    """Initialize the ChessPieceModel and return the model and data loader."""
    model = ChessPieceModel(data_dir="./TrainingImagesPreprocessed")
    return model


@pytest.fixture
def sample_batch(model_and_loader):
    """Fetch a single batch from the train_loader."""
    return next(iter(model_and_loader.train_loader))


def test_data_loading(sample_batch, model_and_loader):
    """Check if data loading works properly."""
    images, labels = sample_batch
    assert images.shape[0] > 0, "DataLoader returned an empty batch!"
    assert len(model_and_loader.train_loader.dataset.classes) > 0, "No classes detected in dataset!"


def test_model_fc_layer(model_and_loader):
    """Ensure model's FC layer is correctly modified for chess classification."""
    num_classes = len(model_and_loader.train_loader.dataset.classes)
    assert model_and_loader.model.fc.out_features == num_classes, "FC layer output does not match class count!"


def test_forward_pass(sample_batch, model_and_loader):
    """Run a forward pass to check model integrity."""
    images, _ = sample_batch
    images = images.to(model_and_loader.device)

    with torch.no_grad():  # Disable gradients for testing
        output = model_and_loader.model(images)

    assert output.shape[0] == images.shape[0], "Model output batch size mismatch!"
    assert output.shape[1] == len(model_and_loader.train_loader.dataset.classes), "Model output class count mismatch!"
