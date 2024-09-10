import unittest
from fastai.vision.all import *
from PIL import Image
from loadandlabel import label_images
from train_model import create_dataloaders_from_image,reload_model_from_checkpoint
from utils import get_dir_path,set_device_to_model

class ChessLabelTest(unittest.TestCase):
    #help function: create a map for image label and image's path.
    def traverse_and_label_images(self, root_dir):
        images_with_labels = []

        # Traverse the directory tree
        for subdir, dirs, files in os.walk(root_dir):
            # Skip the root directory itself
            if subdir == root_dir:
                continue
            for file in files:
                # Check if the file is an image (you can adjust the extensions as needed)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    file_path = os.path.join(subdir, file)
                    # Extract label from the file name (without extension)
                    label = os.path.basename(os.path.dirname(file_path))
                    #label = os.path.splitext(file)[0]
                    # Append tuple (file_path, label) to the list
                    images_with_labels.append((file_path, label))

        return images_with_labels


    def test_chess_label(self):
        """
        Test that chess image matches the model tells.
        """
        model_dir = get_dir_path('models')
        model_file_path = os.path.join(model_dir, 'chess_piece_model.pth')
        training_data_folder = get_dir_path('TrainingImages')
        dls = create_dataloaders_from_image(training_data_folder)
        learn = reload_model_from_checkpoint(model_file_path, dls)
        set_device_to_model(learn)

        # Get the current file's directory (tests/test_some_module.py)

        images_dir = get_dir_path('TestImages')
        image_path_with_labels = self.traverse_and_label_images(images_dir)
        for image_file, label in image_path_with_labels:
            model_predict = self.examine_model(image_file, learn)
            #print(f"image={image_file}, label={label}, predict={model_predict}")
            self.assertEqual(label, model_predict, f"Prediction mismatch-> {image_file}: Expected {label}, Predict {model_predict}")

    def examine_model(self, image_path, learn: Learner):
        img = PILImage.create(image_path)
        pred_class, pred_idx, probs = learn.predict(img)

        class_name = learn.dls.vocab

        if max(probs) < 0.85:
            return "not a chess piece"
        else:
            return class_name[pred_idx]

if __name__ == '__main__':
    unittest.main()
