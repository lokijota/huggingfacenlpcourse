# General notes on chapter 2 - Using Transformers

## Introduction

«The library’s main features are:

- Ease of use: **Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code**.

- Flexibility: At their core, **all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes** and can be handled like any other models in their respective machine learning (ML) frameworks.

- Simplicity: Hardly any abstractions are made across the library. **The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file**, so that the code itself is understandable and hackable.

The tokenizer API is the other main component of the `pipeline()` function. **Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed**. Finally, we’ll show you how to handle sending multiple sentences through a model in a prepared batch, then wrap it all up with a closer look at the high-level `tokenizer()` function.

## Note on Softmax

The softmax function is a mathematical function that converts a vector of real numbers into a vector of probabilities that add up to 1. It is often used in machine learning models, such as neural networks, to perform multi-class classification.

The softmax function works by applying the exponential function to each element of the input vector, and then dividing each element by the sum of all the exponentials. This way, **each element becomes a positive number between 0 and 1, and the whole vector sums to 1**. **The softmax function can be seen as a way of assigning probabilities to different possible outcomes or classes**.

For example, suppose a neural network outputs a vector of three real numbers: (-0.62, 8.12, 2.53). These numbers are not probabilities, and they can be negative or larger than 1. To convert them into probabilities, we can apply the softmax function:

- First, we calculate the exponential (e^x) of each element: (0.54, 3354.73, 12.55).
- Second, we calculate the sum of all the exponentials: 3367.82.
- Third, we divide each element by the sum: (0.00016, 0.9962, 0.0037).

The result is a vector of probabilities: (0.00016, 0.9962, 0.0037). This vector sums to 1, and each element is between 0 and 1. The softmax function has transformed the original vector into a probability distribution that can be interpreted as the confidence of the neural network for each possible class.
