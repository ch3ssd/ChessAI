from PIL import Image
import os

def convert_images_to_rgb(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(img_path)  # Save the converted image

# Use the function to convert all images in your dataset
convert_images_to_rgb('C:/Users/ch3ss/PycharmProjects/ChessAI/OriginalImages')