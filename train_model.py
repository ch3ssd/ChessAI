from fastai.vision.all import *
import matplotlib.pyplot as plt
import os


def get_dir_path(dir_name):
    # Get the current file's directory (tests/test_some_module.py)
    current_file_dir = os.path.dirname(__file__)
    dir_path = os.path.join(current_file_dir, '.', dir_name)
    return dir_path


def load_model(model_dir, model_filename):
    file_path = os.path.join(model_dir, model_filename)
    dls = ImageDataLoaders.from_folder(
        file_path,
        valid_pct=0.3,
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
        bs=16
    )

    learn = vision_learner(dls, resnet18, metrics=accuracy, wd=1e-3)

    checkpoint = torch.load(model_dir / model_filename)
    learn.model.load_state_dict(checkpoint['model_state_dict'])
    learn.opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # Ensure the state of the recorder is set correctly
    learn.recorder.epoch = checkpoint['epoch']
    learn.recorder.losses = checkpoint['loss']

    print(f"Model and optimizer states loaded from {model_dir / model_filename}")
    return learn


def main():
    model_dir = get_dir_path('models')
    model_filename = 'chess_piece_model.pth'

    learn = load_model(model_dir, model_filename)

    learn.fit_one_cycle(6, slice(1e-6, 1e-4), cbs=[EarlyStoppingCallback(patience=4), ReduceLROnPlateau()])

    torch.save({
        'model_state_dict': learn.model.state_dict(),
        'optimizer_state_dict': learn.opt.state_dict(),
        'epoch': learn.recorder.epoch,
        'loss': learn.recorder.losses,
    }, model_dir / model_filename)


if __name__ == '__main__':
    main()
