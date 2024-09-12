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

    images, labels, image_names = label_images(args.path)

    print(f"Loaded {len(images)} from {len(image_names)} images")
    for image_name, label in zip(image_names, labels):
        print(f"Image: {image_name}, Label: {label}")


def label_images(path):
    images = []
    labels = []
    image_names = os.listdir(path)

    for image_name in image_names:
        image_path = os.path.join(path, image_name)
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(image_path)
            if image is not None:
                image = image / 255.0
                images.append(image)
                # Labels should be the specific piece type
                labels.append("king")

    return numpy.array(images), numpy.array(labels), image_names


if __name__ == "__main__":
    main()
