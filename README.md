# ChessAI

ChessAI is a project focused on training a machine learning model to identify chess pieces and their positions based on images of a chessboard. The goal is to recognize various chess pieces and their locations on the board, enabling potential applications such as computer vision-based chess games or automated chess move detection.

This project will be developed iteratively and delivered in phases.

## Phases
1. Train the AI to recognize individual chess pieces from images.
2. Train the AI to identify the position of a single chess piece on the chessboard from images.
3. Train the AI to identify multiple chess pieces and their positions on the chessboard from images.

## Description

This project leverages computer vision and deep learning to classify chess pieces from images of a chessboard. The model is trained to distinguish between various chess pieces such as pawns, knights, rooks, bishops, queens, and kings, as well as identify their colors (white or black).

The project uses frameworks like **PyTorch** and **FastAI** to preprocess the images and train the model. The input consists of chessboard images, and the output will be predictions of the piece type and its color.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

To install the project and its dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ch3ssd/ChessAI.git
   cd ChessAI
(Optional) Create and activate a virtual environment:

2. Make sure that your pip version is 24.2

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
After installing the dependencies, you can start using ChessAI to make predictions on chessboard images.
=======
Training a model to identify chess pieces based off of images of it on a chessboard. 

For Windows/Linux OS
- Install CUDA 12.1, follow the instructions below <br />
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11
1. Choose either Windows/Linux

For Mac:

export PYTORCH_ENABLE_MPS_FALLBACK=1

## Usage
This use of this project is as mentioned earlier in the introduction.
Again, the 3 phases we are aiming to achieve:

Train the AI to:
1. Recognize individual chess pieces from images.
2. Identify the position of a single chess piece on the chessboard from images.
3. Identify multiple chess pieces and their positions on the chessboard from images.

## Dataset

## Training the Model
To train the model:

First, make sure you have all the dependencies installed.
``` bash
pip install -r requirements.txt
```
## Running Tests


## Contributing
If you wish to contribute to this project, here are the steps to get you started:

1. Start by  forking the repository to your github account and clone your forked repository. 
2. Then, create a new branch and you are all set to start working. 
3. After testing your changes and making sure that they work, submit a pull
request and one of the project managers will review your work.

Also, make sure to git pull the updates occasionally.
```bash
   git clone https://github.com/your-username/your-repo.git
 ```
## License
MIT License

Copyright (c) 2024 ch3ssd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
