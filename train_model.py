from fastai.vision.all import *
from utils import get_dir_path, set_device_to_model

def create_dataloaders_from_image(image_folder, valid_pct=0.3):
    """
        Create a single DataLoaders object from multiple image folders.

        :param image_folder: folder path containing images.
        :param valid_pct: Percentage of data to use for validation.
        :return: Combined DataLoaders object.
        """

    print(f"Loading images from '{image_folder}")
    dls = ImageDataLoaders.from_folder(
            image_folder,
            valid_pct=valid_pct,
            item_tfms=Resize(224),
            batch_tfms=aug_transforms(
                mult=2,
                min_scale=0.8,
                max_zoom=1.1,
                flip_vert=False,
                max_rotate=10,
                max_warp=0.1,
                p_lighting=0.3
            ) + [Brightness(p=0.2, max_lighting=0.2),
                Contrast(p=0.2, max_lighting=0.2),
                Saturation(p=0.2, max_lighting=0.2),
                ],
            seed=42,
            bs=16,
            num_workers=0
        )
    return dls


def train_model(dls, model_file_path):
    """
        Train a model using the provided DataLoaders and save the model checkpoint.

        :param dls: DataLoaders object.
        :param model_file_path: Path to save the model checkpoint.
        """
    if not os.path.isfile(model_file_path):
        learn = vision_learner(dls, resnet18, metrics=accuracy, wd=1e-3)
    else:
        learn = reload_model_from_checkpoint(model_file_path, dls)
    # Training the model
    set_device_to_model(learn)
    learn.fit_one_cycle(6, slice(1e-6, 1e-4), cbs=[EarlyStoppingCallback(patience=4), ReduceLROnPlateau()])
    #Save the model checkpoint
    torch.save({
        'model_state_dict': learn.model.state_dict(),
        'optimizer_state_dict': learn.opt.state_dict(),
        'epoch': learn.recorder.epoch,
        'loss': learn.recorder.losses,
    }, model_file_path)
    print(f"Model trained and saved to {model_file_path}")


def reload_model_from_checkpoint(model_file_path, dls):
    """
        Reload the model from a checkpoint.

        :param model_file_path: Path to the model checkpoint file.
        :param dls: DataLoaders object.
        :return: Learner with restored model and optimizer states.
        """
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"The path '{model_file_path}' does not exist")
    checkpoint = torch.load(model_file_path)
    learn = vision_learner(dls, resnet18, metrics=accuracy, wd=1e-3)
    learn.model.load_state_dict(checkpoint['model_state_dict'])
    learn.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    learn.recorder.epoch = checkpoint['epoch']
    learn.recorder.losses = checkpoint['loss']
    print(f"Model and optimizer states loaded from {model_file_path}")
    # You can now use `learn` for further training or inference
    return learn


def main():
    model_dir = get_dir_path('models')
    model_file_path = os.path.join(model_dir, 'chess_piece_model.pth')
    training_data_folder = get_dir_path('TrainingImages')
    dls = create_dataloaders_from_image(training_data_folder)

    #Train model
    train_model(dls, model_file_path)


if __name__ == '__main__':
    main()
