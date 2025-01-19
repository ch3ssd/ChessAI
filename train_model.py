from fastai.vision.all import *
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import os
from collections import defaultdict
import torch

def generate_bootstrap_indices(n_samples, random_state=None):
    """
    Generate bootstrap and Out-Of-Bag (OOB) indices.

    :param n_samples: Total number of samples.
    :param random_state: Seed for reproducibility.
    :return: Tuple of (bootstrap_indices, oob_indices)
    """
    bootstrap_indices = resample(range(n_samples), replace=True, n_samples=n_samples, random_state=random_state)
    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
    return bootstrap_indices, oob_indices

def create_dataloaders(image_folder, train_idxs, valid_idxs=None, bs=16):
    """
    Create DataLoaders with specific training and validation indices.

    :param image_folder: Path to image dataset.
    :param train_idxs: List of training indices.
    :param valid_idxs: List of validation indices (optional).
    :param bs: Batch size.
    :return: DataLoaders object.
    """
    files = get_image_files(image_folder)

    # Create a DataBlock with a custom splitter
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda x: files,
        splitter=IndexSplitter(valid_idxs) if valid_idxs else IndexSplitter([]),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(
            do_flip=True,
            flip_vert=False,
            mult=3,
            max_rotate=10,
            max_zoom=1.2,
            max_warp=0.2,
            p_lighting=0.2,
            p_affine=0.2,
        ) + [
                       Brightness(p=0.2, max_lighting=0.2),
                       Contrast(p=0.2, max_lighting=0.2),
                       Saturation(p=0.2, max_lighting=0.2),
                       RandomErasing(p=0.1, sl=0.02, sh=0.2, min_aspect=0.3, max_count=1)
                   ],
    )

    # Create DataLoaders
    dls = dblock.dataloaders(image_folder, bs=bs, num_workers=0)

    # Assign training items based on train_idxs
    if train_idxs:
        dls.train.items = [files[i] for i in train_idxs]
        print(dls.train.items)

    # Assign validation items based on valid_idxs
    if valid_idxs:
        dls.valid.items = [files[i] for i in valid_idxs]
        print(dls.valid.items)

    return dls

def train_single_model(dls, epochs=5, architecture=resnet18, lr_max=1e-3):
    """
    Train a model on the provided DataLoaders.

    :param dls: DataLoaders object.
    :param epochs: Number of training epochs.
    :param architecture: FastAI model architecture (default: resnet18).
    :param lr_max: Maximum learning rate.
    :return: Trained Learner object.
    """
    learn = vision_learner(dls, architecture, metrics=accuracy, pretrained=True, wd=1e-4, ps=0.2)

    cbs = [
        EarlyStoppingCallback(monitor='valid_loss', patience=3),
        ReduceLROnPlateau(monitor='valid_loss', patience=2, factor=0.2, min_lr=1e-5),
    ]

    learn.fit_one_cycle(epochs, lr_max=lr_max, cbs=cbs)
    return learn

def calculate_bias_variance_confidence_oob(image_folder, n_models=10, oob_epochs=5, ci=95):
    """
    Estimate bias, variance, and calculate confidence interval for accuracy using the OOB technique.

    :param image_folder: Path to image dataset.
    :param n_models: Number of bootstrap models to train.
    :param oob_epochs: Number of epochs for each OOB model.
    :param ci: Confidence interval percentage
    :return: Dictionary with bias, variance, confidence interval for accuracy, and other details.
    """
    files = get_image_files(image_folder)
    n_samples = len(files)
    labels = [parent_label(f) for f in files]
    classes = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_true = np.array([label_to_idx[label] for label in labels])

    # Initialize structures to collect OOB predictions and accuracy metrics
    oob_predictions = defaultdict(list)
    accuracy_list = []

    for model_num in range(n_models):
        print(f"\nTraining OOB model {model_num + 1}/{n_models}")
        bootstrap_idxs, oob_idxs = generate_bootstrap_indices(n_samples, random_state=42 + model_num)

        if not oob_idxs:
            print("No OOB samples for this model, skipping.")
            continue

        # Create DataLoaders for bootstrap sample (training) and OOB samples (validation)
        dls = create_dataloaders(image_folder, train_idxs=bootstrap_idxs, valid_idxs=oob_idxs, bs=16)

        # Train the OOB model
        learn = train_single_model(dls, epochs=oob_epochs, architecture=resnet18, lr_max=1e-3)

        # Get predicted classes for OOB samples
        preds_class, *rest = learn.get_preds(dl=dls.valid, with_decoded=True)
        y_pred = preds_class.argmax(dim=1).cpu().numpy()
        y_true_oob = y_true[oob_idxs]

        # Calculate accuracy for this model
        accuracy = accuracy_score(y_true_oob, y_pred)
        accuracy_list.append(accuracy)

        # Collect predicted probabilities for bias and variance calculation
        preds_probs, *rest = learn.get_preds(dl=dls.valid, with_decoded=False)  # preds shape: (n_oob, n_classes)
        preds_probs = preds_probs.cpu().numpy()

        for idx, pred in zip(oob_idxs, preds_probs):
            oob_predictions[idx].append(pred[y_true[idx]])

        # Clean up to free memory
        del learn
        torch.cuda.empty_cache()

    # Calculate bias and variance
    bias = 0.0
    variance = 0.0
    n_valid = 0

    for idx, preds in oob_predictions.items():
        if not preds:
            continue
        avg_pred = np.mean(preds)
        bias += np.abs(avg_pred - 1)  # Assuming the true probability for the correct class is 1
        variance += np.var(preds)
        n_valid += 1

    avg_bias = bias / n_valid if n_valid else None
    avg_variance = variance / n_valid if n_valid else None

    # Calculate confidence interval for accuracy
    def compute_ci(data, confidence=ci):
        lower = np.percentile(data, (100 - confidence) / 2)
        upper = np.percentile(data, 100 - (100 - confidence) / 2)
        return lower, upper

    accuracy_ci = compute_ci(accuracy_list) if accuracy_list else (None, None)

    print(f"\nBias (approx): {avg_bias:.4f}")
    print(f"Variance (approx): {avg_variance:.4f}")
    if accuracy_ci[0] is not None and accuracy_ci[1] is not None:
        print(f"Accuracy {ci}% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f}")

    return {
        'average_bias': avg_bias,
        'average_variance': avg_variance,
        'accuracy_ci': accuracy_ci,
        'n_models_trained': len(accuracy_list),
        'n_valid_samples': n_valid
    }

def create_dataloaders_from_image(image_folder, valid_pct=0.2, bs=32):
    """
    Create a single DataLoaders object from multiple image folders.

    :param image_folder: folder path containing images.
    :param valid_pct: Percentage of data to use for validation.
    :return: Combined DataLoaders object.
    """
    print(f"Loading images from '{image_folder}'")
    dls = ImageDataLoaders.from_folder(
        image_folder,
        valid_pct=valid_pct,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(
            do_flip=True,
            flip_vert=False,
            mult=3,
            max_rotate=10,
            max_zoom=1.2,
            max_warp=0.2,
            p_lighting=0.2,
            p_affine=0.2,
        ) + [
            Brightness(p=0.2, max_lighting=0.2),
            Contrast(p=0.2, max_lighting=0.2),
            Saturation(p=0.2, max_lighting=0.2),
            RandomErasing(p=0.1, sl=0.02, sh=0.2, min_aspect=0.3, max_count=1)
        ],
        seed=42,
        bs=bs,
        num_workers=0
    )
    return dls

def train_model(dls, model_file_path):
    """
    Train a model using the provided DataLoaders and save the model checkpoint.

    :param dls: DataLoaders object.
    :param model_file_path: Path to save the model checkpoint.
    :param epochs: Number of training epochs.
    """
    checkpoint_path = model_file_path + '.pth'
    if not os.path.isfile(checkpoint_path):
        learn = vision_learner(dls, resnet50, metrics=accuracy, pretrained=True, wd=1e-2, ps=0.8, normalize=True)

        cbs = [
            EarlyStoppingCallback(monitor='valid_loss', patience=15)
        ]

        vals = learn.lr_find()
        lr_max = vals.valley
        learn.fit_one_cycle(2, lr_max=lr_max)

        learn.unfreeze()

        vals_unfrozen = learn.lr_find()
        new_lr_max = vals_unfrozen.valley
        learn.fit_one_cycle(30, lr_max=new_lr_max, cbs=cbs)

        # Save the trained model
        learn.save('chess_piece_model')
        print(f"Model trained and saved to {checkpoint_path}")
    else:
        print("Reloading model for continued training: ")
        training_data_folder = r"G:\My Drive\ChessAIProject\ChessPieceImages\TrainingImagesNoBG"
        learn = reload_model_from_checkpoint(model_file_path, training_data_folder, valid_pct=0.2, bs=32)

        learn.fit_one_cycle(2, lr_max=1e-3)
        learn.save('chess_piece_model')
        print(f"Model trained and saved to {checkpoint_path}")

    # Ensure model and data are on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn.model.to(device)
    dls.device = device

    return learn

def reload_model_from_checkpoint(model_file_path, image_folder, valid_pct=0.2, bs=32):
    """
    Reload the model from a checkpoint.

    :param model_file_path: Path to the model checkpoint file.
    :return: Learner with restored model and optimizer states.
    """
    checkpoint_path = model_file_path + '.pth'
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"The path '{checkpoint_path}' does not exist")

    #Recreate Dataloaders
    print("Recreating DataLoaders for continued training...")
    dls = create_dataloaders_from_image(image_folder, valid_pct=valid_pct, bs=bs)

    learn = vision_learner(
        dls,
        resnet50,
        metrics=accuracy,
        pretrained=True,
        wd=1e-2,
        ps=0.8,
        normalize=True
    )

    learn.load('chess_piece_model')
    print(f"Model weights loaded from '{checkpoint_path}'")

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn.model.to(device)
    learn.dls.device = device
    print(f"Model and DataLoaders moved to {device}.")

    return learn

# Main Function

def main():
    model_dir = r"G:\My Drive\ChessAIProject\models"
    training_data_folder = r"G:\My Drive\ChessAIProject\ChessPieceImages\TrainingImagesNoBG"

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Create DataLoaders for the main model
    dls = create_dataloaders_from_image(training_data_folder, valid_pct=0.2, bs=32)

    # Define model file path
    model_file_path = os.path.join(model_dir, 'chess_piece_model')

    # Train the main model
    print("Training the main model...")
    train_model(dls, model_file_path)

    # Calculate Bias, Variance, and Confidence Interval for Accuracy using OOB technique
    print("\nCalculating Bias, Variance, and Confidence Interval for Accuracy using Out-of-Bag (OOB) technique...")
    bias_variance_ci = calculate_bias_variance_confidence_oob(
        training_data_folder,
        n_models=10,
        oob_epochs=5,
        ci=95  # Confidence interval percentage
    )

    # Display Results
    print("\nBias-Variance and Confidence Interval Analysis (OOB):")
    for key, value in bias_variance_ci.items():
        if key == 'accuracy_ci' and value is not None:
            print(f"{key}: {value[0]:.4f} - {value[1]:.4f}")
        else:
            print(f"{key}: {value}")

# Execute the script
if __name__ == '__main__':
    main()
