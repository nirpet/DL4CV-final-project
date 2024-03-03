import torchvision
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from keras.layers import Dense, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import foolbox as fb
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from inference import convert_torch_to_tf
import torch

import inference


def train_and_save_model_on_MNIST_dataset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_train = train_images.reshape(60000, 28, 28, 1) / 255
    x_test = test_images.reshape(10000, 28, 28, 1) / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.save('/models/model_for_MNIST_dataset.keras')


# print('start')
# train_and_save_model_on_MNIST_dataset()
# print('end')

def save_resnet50_model():
    # Load the ResNet50 model pre-trained on ImageNet
    model = ResNet50(weights='imagenet')
    model2 = torchvision.models.resnet18(pretrained=True)


def test_fool_box():
    model = torchvision.models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
    print(labels)
    # print(fb.utils.accuracy(fmodel, images, labels))
    attack = fb.attacks.LinfDeepFoolAttack()
    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
    print(is_adv)
    #
    # model.eval()
    #
    # # Make a prediction
    # with torch.no_grad():
    #     outputs = model(clipped)
    #
    # probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    # _, predicted_idx = torch.max(probabilities, 0)
    # print(predicted_idx)
    image_to_display = (clipped[0]).squeeze().permute(1, 2, 0)
    image_to_display = image_to_display.clamp(0, 1)
    plt.imshow(image_to_display)
    plt.axis('off')
    plt.show()


def fgsm_attack(image, epsilon, gradient):
    # Get the sign of the gradient
    signed_grad = tf.sign(gradient)
    # Perturb the image
    perturbed_image = image + epsilon * signed_grad
    # Make sure the image remains in the [0,1] range
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    return perturbed_image


def generate_attack(model, epsilon=0.15):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_test = test_images.reshape(10000, 28, 28, 1) / 255
    y_test = to_categorical(test_labels)
    original_predicted_class = 0
    adversarial_prediction_no_defence_class = 0
    iterations = 0
    perturbed_image = 0
    random_image = x_test[0]

    # Select a random image from the test dataset
    while original_predicted_class == adversarial_prediction_no_defence_class:
        random_index = np.random.randint(0, len(x_test))
        random_image = x_test[random_index]

        # Assume `image` is preprocessed and expanded to have batch dimension [1, 28, 28]
        # `label` should also be in a shape that the model expects - for instance, [1,] for a single scalar label.
        image = tf.convert_to_tensor(random_image)  # Convert to tf.Tensor
        image = tf.expand_dims(image, 0)  # Add batch dimension

        # Predict the class of the random image
        original_prediction = model.predict(np.array([random_image]), verbose=0)  # Model expects a batch of images
        original_predicted_class = np.argmax(original_prediction, axis=1)

        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(original_predicted_class, prediction,
                                                                   from_logits=False)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        # Get the sign of the gradients to create the perturbation
        perturbed_image = fgsm_attack(image, epsilon, gradient)
        logits = model(perturbed_image)
        adversarial_prediction_no_defence = tf.argmax(logits, axis=1)
        adversarial_prediction_no_defence_class = adversarial_prediction_no_defence.numpy()[0]

        iterations += 1
    return random_image, perturbed_image, original_predicted_class, adversarial_prediction_no_defence_class


def print_results(original_image, advesarial_image, original_predicted_class, adversarial_prediction_no_defence_class,
                  adversarial_prediction_with_defence_class):
    print(f"Original image Model prediction: {original_predicted_class[0]}", )
    print("Predicted label for adversarial NO DEFENCE", adversarial_prediction_no_defence_class)
    print("Predicted label for adversarial WITH DEFENCE", adversarial_prediction_with_defence_class)

    # random_image_to_show = random_image.reshape(28, 28)  # Reshape for plotting
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    adversarial_perturbed_image_to_show = advesarial_image.reshape(28, 28)
    original_image_to_show = original_image.reshape(28, 28)  # Reshape for plotting

    # Display the image
    plt.imshow(adversarial_perturbed_image_to_show, cmap='gray')
    plt.title("adversarial perturbed image")
    plt.show()

    plt.imshow(original_image_to_show, cmap='gray')
    plt.title("Original image")
    plt.show()
    print("Win" if original_predicted_class == adversarial_prediction_with_defence_class else "Lose")
    return original_predicted_class == adversarial_prediction_with_defence_class


model_path = '/models/model_for_MNIST_dataset.keras'
model = load_model(model_path)


def success_defence_rate(iters=50, num_of_samples=10, noise_range=140, epsilon=0.2):
    success_defence_counter = 0
    no_defence_counter = 0
    true_label_confidence_rate = 0
    for i in range(iters):
        original_image, adversarial_image, original_predicted_class, adversarial_predicted_no_defence_class = generate_attack(
            model, epsilon)
        adversarial_predicted_with_defence, confidence_rate = inference.predict(model, adversarial_image,
                                                                                num_of_samples=num_of_samples,
                                                                                noise_range=noise_range,
                                                                                visualize=False)
        adversarial_predicted_with_defence_class = np.argmax(adversarial_predicted_with_defence.numpy())
        success_defence_counter += 1 if adversarial_predicted_with_defence_class == original_predicted_class else 0
        true_label_confidence_rate += confidence_rate[original_predicted_class]
        no_defence_counter += 1 if adversarial_predicted_no_defence_class == original_predicted_class else 0
    success_defence_rate = success_defence_counter / iters
    true_label_confidence_rate = true_label_confidence_rate / iters
    return success_defence_rate, true_label_confidence_rate


def success_rate_of_predicting_original_with_defence(iters=100, num_of_samples=10, noise_range=140, epsilon=0.2,
                                                     visualize=False):
    success_defence_counter = 0
    for i in range(iters):
        original_image, advers, original_predicted_class, _ = generate_attack(
            model, epsilon)
        original_image = torch.unsqueeze(torch.tensor(original_image), dim=0)
        original_image = convert_torch_to_tf(original_image)
        # print(original_image.shape)
        original_predicted_with_defence = inference.predict(model, original_image, num_of_samples=num_of_samples,
                                                            noise_range=noise_range, visualize=visualize)
        original_predicted_with_defence_class = np.argmax(original_predicted_with_defence.numpy())
        success_defence_counter += 1 if original_predicted_with_defence_class == original_predicted_class else 0
    success_defence_rate = success_defence_counter / iters
    return success_defence_rate


def success_as_function_of_noise_rage(noise_ranges):  # Define noise range values to explore

    # Calculate success defense rates for each noise range
    success_defence_rates = [success_defence_rate(noise_range=noise_range) for noise_range in noise_ranges]
    # success_defence_rates = [success_defence_rate(num_of_samples=num_of_samples) for num_of_samples in num_of_samples_ranges]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(noise_ranges, success_defence_rates, marker='o', linestyle='-', color='blue')
    # plt.plot(success_defence_rates, success_defence_rates, marker='o', linestyle='-', color='blue')
    plt.title('Success Defense Rate as a Function of Noise Range')
    plt.xlabel('Noise Range')
    plt.ylabel('Success Defense Rate')
    plt.grid(True)
    plt.show()


def success_as_function_of_num_of_samples(num_of_samples_ranges, noise_range=50,
                                          iters=50):  # Define noise range values to explore

    success_defence_rates = [
        success_defence_rate(num_of_samples=num_of_samples, noise_range=noise_range, iters=iters, epsilon=0.15)[0] for
        num_of_samples in
        num_of_samples_ranges]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(num_of_samples_ranges, success_defence_rates, marker='o', linestyle='-', color='blue')
    plt.title('Success Defense Rate as a Function of Noise Range')
    plt.xlabel('Samples Range')
    plt.ylabel('Success Defense Rate')
    plt.grid(True)
    plt.show()


def success_of_original_image_as_function_of_noise_range(noise_ranges, iters=50):
    success_defence_rates = [success_rate_of_predicting_original_with_defence(noise_range=noise_range, iters=iters) for
                             noise_range in noise_ranges]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(noise_ranges, success_defence_rates, marker='o', linestyle='-', color='blue')
    plt.title('Success Defense Rate as a Function of Noise Range')
    plt.xlabel('Noise Range')
    plt.ylabel('Success Defense Rate')
    plt.grid(True)
    plt.show()


noise_ranges = np.arange(30, 70, 7)
num_of_samples_ranges = np.arange(1, 10, 2)
success_as_function_of_num_of_samples(num_of_samples_ranges=num_of_samples_ranges, noise_range=110, iters=100)

# def success_of_low_pass_as_function_of_noise_range(noise_ranges, iters=50):
#     success_defence_rates = [success_defence_rate_and_apply_low_pass_filter(noise_range=noise_range, iters=iters) for
#                              noise_range in noise_ranges]
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(noise_ranges, success_defence_rates, marker='o', linestyle='-', color='blue')
#     # plt.plot(success_defence_rates, success_defence_rates, marker='o', linestyle='-', color='blue')
#     plt.title('Success of low-pass as a Function of Noise Range')
#     plt.xlabel('Noise Range')
#     plt.ylabel('Success Defense Rate')
#     plt.grid(True)
#     plt.show()

# def success_defence_rate_and_apply_low_pass_filter(iters=50, num_of_samples=10, visualize=False, noise_range=140,
#                                                    epsilon=0.2):
#     success_defence_counter = 0
#     for i in range(iters):
#         original_image, advesarial_image, original_predicted_class, adversarial_predicted_no_defence_class = generate_attack(
#             model, epsilon)
#         adversarial_predicted_with_defence = inference.predict(model, advesarial_image, num_of_samples=num_of_samples,
#                                                                noise_range=noise_range, visualize=visualize)
#         adversarial_predicted_with_defence_class = np.argmax(adversarial_predicted_with_defence.numpy())
#         success_defence_counter += 1 if adversarial_predicted_with_defence_class == original_predicted_class else 0
#     success_defence_rate = success_defence_counter / iters
#     return success_defence_rate
