# Neural Network from Scratch (NumPy)

I got super bored and decided to build a neural network model from scratch using numpy.

The project is a WIP and is being used as a side project during block 4 of ISE

I've been meaning to code in python more often, and what this does is predict student pass/fail outcomes based on study hours.

I also impliment just the core ideas of an nn ie.
- Forward propagation
- Loss calculation
- Back propagation
- Gradient descent

### Forward Pass
So far I've added a forward pass function that takes the input and is multiplied by the weight and bias is added:
Z = Wx + b

This is passed through a sigmoid function to produce a probability between 0 and 1.

## Dataset

As seen in main, I used just a normal array of numbers which is used as a synthetic dataset.

| Hours Studied | Result |
|--------------|--------|
| 1            | 0 (Fail) |
| 2            | 0 |
| 3            | 0 |
| 4            | 1 (Pass) |
| 5            | 1 |

The goal is for the model to learn the threshold where a student transitions from failing to passing.

### Goals of the project
- For me to gain a better grasp on python and how neural networks work on a larger scale.

## How to Run

1. Install dependencies:
pip install numpy matplotlib

2. Run the script:
python nn_from_scratch.py