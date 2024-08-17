from fastai.vision.all import *


def load_model(model_dir, model_filename):
    dls = ImageDataLoaders.from_folder(
        Path('C:/Users/ch3ss/PycharmProjects/ChessAI/TrainingImages'),
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

    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # Load the saved weights into the model
    checkpoint = torch.load(model_dir / model_filename)
    learn.model.load_state_dict(checkpoint['model_state_dict'])
    learn.opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # Ensure the state of the recorder is set correctly
    learn.recorder.epoch = checkpoint['epoch']
    learn.recorder.losses = checkpoint['loss']

    print(f"Model and optimizer states loaded from {model_dir / model_filename}")
    return learn

def test_model(learn, image_path):
    img = PILImage.create(image_path)
    pred_class, pred_idx, probs = learn.predict(img)

    class_names = learn.dls.vocab
    print(probs)
    if max(probs) < 0.85:
        return "not a chess piece"
    else:
        return class_names[pred_idx]

def main():
    model_dir = Path('C:/Users/ch3ss/PycharmProjects/ChessAI/models')
    model_filename = 'chess_piece_model.pth'

    learn = load_model(model_dir, model_filename)

    # Path to the image you want to test
    test_image_path = Path('C:/Users/ch3ss/PycharmProjects/ChessAI/TestImages/test_king6.jpg')
    result = test_model(learn, test_image_path)
    print(f'Result: {result}')

if __name__ == '__main__':
    main()
