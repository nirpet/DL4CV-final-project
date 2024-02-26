import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_images(images, num_rows=2, num_cols=5):
    """
    Visualize a grid of images.

    Args:
    - images (torch.Tensor): Tensor containing images to visualize.
    - num_rows (int): Number of rows in the grid.
    - num_cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(images):
                axes[i, j].imshow(images[index].numpy().reshape(28, 28))
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def generate_noisy_images(image, num_of_samples):
    """
    Generate noisy samples of an input image.

    Args:
    - image (torch.Tensor): Input image tensor.
    - num_of_samples (int): Number of noisy samples to generate.

    Returns:
    - noisy_images (torch.Tensor): Tensor containing noisy samples of the input image.
    """
    # Initialize a tensor to store noisy samples
    shape = list(image.shape)
    shape[0] = num_of_samples
    noisy_images = torch.empty(tuple(shape))
    noise_range = 100

    # Gaussian noise
    # std_dev = 10

    # Loop to generate noisy samples
    for i in range(num_of_samples):
        # Gaussian noise
        # noise = torch.randn(image.shape) * std_dev
        noise = torch.randint(-noise_range, noise_range + 1, size=tuple(image.shape))/255
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        noisy_images[i] = noisy_image

    return noisy_images

def predict(model, input_image, num_of_samples=10, visualize=False):
    """
    Perform inference using the loaded model on noisy samples of an input image,
    and return the majority vote prediction.

    Args:
    - model: The pre-trained neural network model.
    - input_image (torch.Tensor): Input image tensor.
    - num_of_samples (int): Number of noisy samples to generate. Default is 10.
    - visualize (bool): Whether to visualize the noisy images. Default is False.

    Returns:
    - majority_vote_prediction (torch.Tensor): Majority vote prediction in one-hot format.
    """
    input_data = convert_tf_to_torch(input_image)
    noisy_images = generate_noisy_images(input_data, num_of_samples)

    if visualize:
        visualize_images(noisy_images)

    predictions = torch.empty((len(noisy_images),10))
    with torch.no_grad():
        for i in range(noisy_images.shape[0]):
            prediction = convert_tf_to_torch(model(convert_torch_to_tf(noisy_images[i].unsqueeze(0))))
            predictions[i] = prediction.squeeze()

    # Discretize predictions to one-hot vectors
    one_hot_predictions = torch.zeros_like(predictions)
    max_indices = torch.argmax(predictions, dim=1)
    one_hot_predictions.scatter_(1, max_indices.unsqueeze(1), 1)

    # Calculate the mode (most common value) of the one-hot predictions
    majority_vote_prediction = torch.mode(one_hot_predictions, dim=0)[0]



    return majority_vote_prediction

def convert_tf_to_torch(tf_input):
    np_image = tf_input.numpy()

    # Step 2: Convert NumPy array to PyTorch tensor
    return torch.from_numpy(np_image)

def convert_torch_to_tf(torch_input):
    # Assuming torch_tensor is your PyTorch tensor
    # Step 1: Move tensor to CPU if it's on GPU, then convert it to NumPy array
    torch_tensor_cpu = torch_input.cpu().numpy()

    # Step 2: Convert NumPy array to TensorFlow tensor
    tensor_flow = tf.convert_to_tensor(torch_tensor_cpu)
    return tensor_flow