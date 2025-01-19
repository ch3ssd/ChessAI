from fastai.vision.all import *
from pathlib import Path
from stat_tests import calculate_entropy, classify_entropy

def load_model(model_dir, model_filename):
    """
    Load the trained Learner using FastAI's load_learner method.

    :param model_dir: Directory where the model file is located.
    :param model_filename: Full filename of the exported Learner (including extension).
    :return: Loaded Learner object.
    """
    model_pkl_path = Path(model_dir) / model_filename

    if not model_pkl_path.exists():
        raise FileNotFoundError(f"The model file '{model_pkl_path}' does not exist.")

    learn = load_learner(model_pkl_path)
    print(f"Model loaded successfully from '{model_pkl_path}'")
    return learn

def predict_and_classify(learn, image_path, entropy_threshold):
    """
    Predict the class of a single chess piece image and classify using entropy-based measure.

    :param learn: Loaded Learner object.
    :param image_path: Path to the image file.
    :param entropy_threshold: Entropy threshold for the test.
    :return: Dictionary with image name, final label, probabilities, and entropy.
    """
    try:
        img = PILImage.create(image_path)
        pred_class, pred_idx, probs = learn.predict(img)
        probs_list = [float(p) for p in probs]

        # Perform entropy-based classification to assign Final_Label
        final_label, entropy = classify_entropy(probs_list, learn.dls.vocab, entropy_threshold)
        class_prob_dict = dict(zip(learn.dls.vocab, probs_list))

        return {
            'Image': image_path.name,
            'Final_Label': final_label,
            'Probabilities': class_prob_dict,
            'Entropy': entropy
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {
            'Image': image_path.name,
            'Final_Label': "Error",
            'Probabilities': [],
            'Entropy': None
        }

def process_directory(learn, test_images_dir, entropy_threshold):
    """
    Process all images in the directory and perform entropy-based classification.

    :param learn: Loaded Learner object.
    :param test_images_dir: Path to the test images directory.
    :param entropy_threshold: Entropy threshold for the test.
    :return: List of prediction results.
    """
    predictions = []

    for sub_dir in test_images_dir.iterdir():
        if sub_dir.is_dir():
            for image_file in sub_dir.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.png', '.jpeg', '.JPG']:
                    result = predict_and_classify(learn, image_file, entropy_threshold=entropy_threshold)
                    predictions.append(result)
                    # Print detailed results with Final_Label
                    if result['Entropy'] is not None:
                        print(
                            f"Image: {result['Image']}, "
                            f"Final Label: {result['Final_Label']}, "
                            f"Probabilities: {result['Probabilities']}, "
                            f"Entropy: {result['Entropy']:.2f}"
                        )
                    else:
                        print(
                            f"Image: {result['Image']}, "
                            f"Final Label: {result['Final_Label']}, "
                            f"Probabilities: {result['Probabilities']}, "
                            f"Entropy: N/A"
                        )

    return predictions

def main():
    """
    Main function to orchestrate the testing process.
    """
    # --- Use your custom paths directly here ---
    model_dir = r"G:\My Drive\ChessAIProject\models"
    test_images_dir = r"G:\My Drive\ChessAIProject\ChessPieceImages\TestImages"

    model_filename = 'chess_piece_model.pkl'  # Matches your saved model name

    model_dir = Path(model_dir)
    test_images_dir = Path(test_images_dir)

    learn = load_model(model_dir, model_filename)

    entropy_threshold = 0.7

    """
    predictions = process_directory(
        learn,
        test_images_dir,
        entropy_threshold=entropy_threshold
    )
    """

    #To test a single image

    test_image_path = Path('G:/My Drive/ChessAIProject/ChessPieceImages/TestImages/Images/IMG_0774.JPG')
    print(predict_and_classify(learn,test_image_path,entropy_threshold))



if __name__ == '__main__':
    main()
