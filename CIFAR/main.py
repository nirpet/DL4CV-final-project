import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from CIFAR.cifar_model import *
from CIFAR.attack import *
from CIFAR.config import *

from CIFAR.src.train import Trainer
from CIFAR.src.eval import Evaluator
import json

def main():
    args = parse_args()
    configs = get_configs(args)

    model = WRN(depth=configs.model_depth, width=configs.model_width, num_classes=configs.num_class)
    #model = Normalize_net(model) # apply the normalization before feeding the inputs into the classifier.
    
    if configs.mode == 'train':
        train = Trainer(configs, model)
        train.train_model()
    elif configs.mode == 'eval':
        test = Evaluator(configs, model)
        test.eval_model()
    else:
        raise ValueError('Specify the mode, `train` or `eval`')


if __name__ == '__main__':
    main()