import os
import torch
from fastai.learner import Learner
def get_dir_path(dir_name):
    # Get the current file's directory (tests/test_some_module.py)
    current_file_dir = os.path.dirname(__file__)
    dir_path = os.path.join(current_file_dir, '.', dir_name)
    return dir_path

# Determine the device (CUDA/MPS/CPU)
def determine_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
    print(f"device = {device}")
    return device

def set_device_to_model(learn: Learner):
    if determine_device() == "cpu":
        learn.dls.cpu()
    else:
        learn.dls.cuda()