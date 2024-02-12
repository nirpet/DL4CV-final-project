import random

import torchvision

path = '.'

def get_CIFAR10(path=path, train=True, download=True):
    CIFAR10 = torchvision.datasets.CIFAR10(root=path, train=train, download=download)
    return CIFAR10

def get_MNIST(path=path, train=True, download=True):
    MNIST = torchvision.datasets.MNIST(root=path, train=train, download=download)
    return MNIST

def show_image(dataset, index):
    image = dataset[index][0]
    tag = dataset[index][1]
    print(f'showing image of tag={tag}')
    image.show()

def main():
    MNIST = get_MNIST()
    CIFAR10 = get_CIFAR10()

    random_index = random.randint(0, 100)

    show_image(MNIST, random_index)
    show_image(CIFAR10, random_index)

main()
