import unittest
import os
from PIL import Image
from loadandlabel import label_images


class ChessLabelTest(unittest.TestCase):
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
        # Get the current file's directory (tests/test_some_module.py)
        current_file_dir = os.path.dirname(__file__)
        images_dir = os.path.join(current_file_dir, '..', 'TestImages')
        image_with_labels = self.traverse_and_label_images(images_dir)
        print(image_with_labels)

        #images, labels, image_names = label_images(directory)

        #self.assertEqual(len(images), len(image_names))
        #self.assertEqual(len(labels), len(image_names))

        #for i, image_name in enumerate(image_names):
        #    expected_label = "king"
        #    self.assertEqual(labels[i], expected_label)


if __name__ == '__main__':
    unittest.main()
