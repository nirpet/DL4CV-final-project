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
import torch
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

    model.save('/model_for_MNIST_dataset')

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


def test_gpt_attack():
    model_path = '/Users/michaelsieradzki/PycharmProjects/DL4CV-final-project/my_model_savedmodel'

    # Load the model
    model = load_model(model_path)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_test = test_images.reshape(10000, 28, 28, 1) / 255
    y_test = to_categorical(test_labels)

    # Select a random image from the test dataset
    random_index = np.random.randint(0, len(x_test))
    random_image = x_test[random_index]

    # Assume `image` is preprocessed and expanded to have batch dimension [1, 28, 28]
    # `label` should also be in a shape that the model expects - for instance, [1,] for a single scalar label.
    image = tf.convert_to_tensor(random_image) # Convert to tf.Tensor
    image = tf.expand_dims(image, 0)  # Add batch dimension

    # Predict the class of the random image
    prediction = model.predict(np.array([random_image]))  # Model expects a batch of images
    predicted_class = np.argmax(prediction, axis=1)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(predicted_class, prediction, from_logits=False)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    # Get the sign of the gradients to create the perturbation
    epsilon = 0.15  # Small enough to be imperceptible, yet large enough to cause misclassification
    perturbed_image = fgsm_attack(image, epsilon, gradient)
    logits = model(perturbed_image)
    prediction = tf.argmax(logits, axis=1)
    print("Predicted label for adversarial example:", prediction.numpy())

    # random_image_to_show = random_image.reshape(28, 28)  # Reshape for plotting
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    random_image_to_show = perturbed_image.reshape(28, 28)  # Reshape for plotting

    # Display the image
    plt.imshow(random_image_to_show, cmap='gray')
    plt.title("Random Image from Test Set")
    plt.show()



    print(f"Model prediction: {predicted_class[0]}", )
    print(prediction)

    # num_examples = 50
    # subset_indices = np.random.choice(x_test.shape[0], num_examples, replace=False)
    # x_test_subset = x_test[subset_indices]
    # y_test_subset = y_test[subset_indices]
    #
    # # Generate adversarial examples for the subset
    # # Note: This assumes `ifgsm_attack` is defined and ready to use
    # x_test_adv_subset = np.array(
    #     [fgsm_attack(image, epsilon).numpy() for image in x_test_subset])
    #
    # # Ensure data shape compatibility with the model if needed
    # # This step might vary depending on your model's expected input shape
    # x_test_adv_subset = x_test_adv_subset.reshape(num_examples, 28, 28)
    #
    # # Evaluate the model on the original subset
    # loss_orig, accuracy_orig = model.evaluate(x_test_subset, y_test_subset, verbose=0)
    # print(f"Accuracy on Original Examples: {accuracy_orig * 100:.2f}%")
    #
    # # Evaluate the model on the adversarial subset
    # loss_adv, accuracy_adv = model.evaluate(x_test_adv_subset, y_test_subset, verbose=0)
    # print(f"Accuracy on Adversarial Examples: {accuracy_adv * 100:.2f}%")


# test_fool_box()
test_gpt_attack()








