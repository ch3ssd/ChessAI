import os
import cv2
import numpy

def load_and_label_images(base_path):
    images = []
    labels = []
    class_names = os.listdir(base_path)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(base_path,class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg','.jpeg','.png')):
                    image_path = os.path.join(class_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = image/255.0
                        images.append(image)
                        labels.append(label)

    return numpy.array(images), numpy.array(labels), class_names

if __name__ == "__main__":
    base_path = 'ChessPieces Images/Dataset'
    images, labels, class_names = load_and_label_images(base_path)
    print(f"Loaded {len(images)} from {len(class_names)} classes")
    print(f"Class names {class_names}")