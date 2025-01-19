import numpy as np

def calculate_entropy(probs):
    """
    Calculate the normalized entropy of a probability distribution in bits.
    :param probs: A list or NumPy array of probabilities for each class.
    :return: Normalized entropy value between 0 and 1.
    """
    probs = np.array(probs)
    epsilon = 1e-10  # To prevent log(0) issues
    entropy = -np.sum(probs * np.log2(probs + epsilon))
    max_entropy = np.log2(len(probs))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def classify_entropy(probs, vocab, entropy_threshold):
    """
    Classify the prediction based on entropy.

    :param probs: A list or NumPy array of probabilities for each class.
    :param vocab: List of class labels corresponding to probabilities.
    :param entropy_threshold: The entropy threshold below which the prediction is considered confident.
    :return: Selected label if entropy is below the threshold, else "Uncertain".
    """
    entropy = calculate_entropy(probs)
    if entropy < entropy_threshold:
        top_idx = np.argmax(probs)
        selected_label = vocab[top_idx]
        return selected_label, entropy
    else:
        return "Uncertain", entropy
