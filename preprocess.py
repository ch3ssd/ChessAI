import cv2
import numpy
import os

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image Not Found: {image_path}")

    image = cv2.resize(image,target_size)

    image = image/255.0
    image_array = numpy.array(image)

    return image_array

def preprocess_and_save_images(input_directory,output_directory,target_size = (450,450)):
    preprocessed_images = []
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg','.jpeg','.png')):
            image_path = os.path.join(input_directory,filename)
            try:
                preprocessed_image = preprocess_image(image_path,target_size)
                save_path = os.path.join(output_directory,filename)
                cv2.imwrite(save_path,preprocessed_image * 255.0)

            except ValueError:
                print("error")

#Example
input_directory = 'ChessPieces Images/WhiteBishop'
output_directory = 'ChessPieces Images/WhiteBishopProcessed'

preprocess_and_save_images(input_directory,output_directory)


