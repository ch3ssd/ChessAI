import os
import cv2
import numpy
import argparse

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Load images and labels from a specified file path and print them")

    # Add arguments
    parser.add_argument('path', type=str, help='Path of images')

    # Parse the arguments
    args = parser.parse_args()

    images = []
    labels = []
    image_names = os.listdir(args.path)

    for label, image_name in enumerate(image_names):
        image_path = os.path.join(args.path, image_name)
        if image_path.endswith(('.jpg','.jpeg','.png')):
           image = cv2.imread(image_path)
           if image is not None:
              image = image/255.0
              images.append(image)
              #TODO the label shall be a notation instead of a sequence num
              labels.append(label)

    return numpy.array(images), numpy.array(labels), image_names
if __name__ == "__main__":
    images, labels, image_names = main()
    print(f"Loaded {len(images)} from {len(image_names)} images")
    for image_name, label in zip(image_names, labels):
        print(f"Image: {image_name}, Label: {label}")
