import unittest
import os
from loadandlabel import label_images

class MyTestCase(unittest.TestCase):
    def test_king_label(self):
        directory = "king"
        image_path = os.path.join(directory, "king1.png")

        images, labels, image_names = label_images(directory)

        self.assertEqual(len(images), len(image_names))
        self.assertEqual(len(labels), len(image_names))

        for i, image_name in enumerate(image_names):
            expected_label = "king"
            self.assertEqual(labels[i], expected_label)

if __name__ == '__main__':
    unittest.main()
