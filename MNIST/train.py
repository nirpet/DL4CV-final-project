import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from inference import predict, predict_experiment

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore",
                        message="dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_MODEL_PATH = 'MNIST/models/base_model.pth'
F_MODEL_PATH = 'MNIST/models/f_model.pth'
F1_MODEL_PATH = 'MNIST/models/f1_model.pth'

np.random.seed(42)
torch.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
dataset = datasets.MNIST(root='./MNIST/data', train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
test_set = datasets.MNIST(root='./MNIST/data', train=False, transform=transform, download=True)
test_set = torch.utils.data.Subset(test_set, range(500))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def base_fit(model, device, train_loader, val_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    data_loader = {'train': train_loader, 'val': val_loader}
    print("Fitting the model...")
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        loss_per_epoch, val_loss_per_epoch = 0, 0
        for phase in ('train', 'val'):
            for i, data in enumerate(data_loader[phase]):
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                # calculating loss on the output
                loss = criterion(output, label)
                if phase == 'train':
                    optimizer.zero_grad()
                    # grad calc w.r.t Loss func
                    loss.backward()
                    # update weights
                    optimizer.step()
                    loss_per_epoch += loss.item()
                else:
                    val_loss_per_epoch += loss.item()
        scheduler.step(val_loss_per_epoch / len(val_loader))
        print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch + 1, loss_per_epoch / len(train_loader),
                                                       val_loss_per_epoch / len(val_loader)))
        train_loss.append(loss_per_epoch / len(train_loader))
        val_loss.append(val_loss_per_epoch / len(val_loader))
    return model, train_loss, val_loss


def training_results(loss, val_loss):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.arange(1, 11), loss, "*-", label="Loss")
    plt.plot(np.arange(1, 11), val_loss, "o-", label="Val Loss")
    plt.xlabel("Num of epochs")
    plt.legend()
    plt.show()


def experiment(model, data, perturbed_data, adv_pred, target_pred):
    with torch.no_grad():# Create a figure with two subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 3]})

        # Plot the original image in the first subplot
        axs[0].imshow(data.reshape((28,28)), cmap="gray")
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Plot the perturbed image in the second subplot
        axs[1].imshow(perturbed_data.reshape((28,28)), cmap="gray")
        axs[1].set_title('Perturbed Image')
        axs[1].axis('off')

        noise_range = [0, 10, 30, 50, 70, 100, 150, 200]
        num_of_samples = 500
        orig_accuracies = []
        adv_accuracies = []
        adv_conf_accuracies = []
        orig_pred_probs = []
        adv_pred_probs = []
        for noise in noise_range:
            orig_vicinity = predict_experiment(model, data, num_of_samples, noise)
            adv_vicinity = predict_experiment(model, perturbed_data, num_of_samples, noise)

            orig_prob = torch.sum(orig_vicinity, dim=0) / num_of_samples
            adv_prob = torch.sum(adv_vicinity, dim=0) / num_of_samples

            orig_accuracy = len(torch.where((orig_vicinity == nn.functional.one_hot(torch.tensor(target_pred), num_classes=10)).all(dim=1))[0]) / num_of_samples
            orig_accuracies.append(orig_accuracy)
            adv_accuracy = len(torch.where((adv_vicinity == nn.functional.one_hot(torch.tensor(target_pred), num_classes=10)).all(dim=1))[0]) / num_of_samples
            adv_conf_accuracy = len(torch.where((adv_vicinity == nn.functional.one_hot(torch.tensor(int(adv_pred)), num_classes=10)).all(dim=1))[0]) / num_of_samples
            adv_accuracies.append(adv_accuracy)
            adv_conf_accuracies.append(adv_conf_accuracy)
            orig_pred_probs.append(orig_prob)
            adv_pred_probs.append(adv_prob)

        axs[2].plot(noise_range, orig_accuracies, "*-", label="Original Image")
        axs[2].plot(noise_range, adv_accuracies, "o-", label="Perturbed Image - Correct")
        axs[2].plot(noise_range, adv_conf_accuracies, "x-", label="Perturbed Image - Adversarial")
        axs[2].set_title('Accuracy in the Vicinity of the Image')
        axs[2].set_xlabel("Noise Range")
        axs[2].set_ylabel("Accuracy")
        axs[2].legend()
        axs[2].grid(True)
        # plt.tight_layout()
        plt.show()

    bar_width = 0.3
    for idx, noise in enumerate(noise_range):
        plt.figure(figsize=(8, 5))
        index = np.arange(10)
        orig_bars = plt.bar(index + bar_width / 2, orig_pred_probs[idx], bar_width, color='b', label='Original Image')
        adv_bars = plt.bar(index + 3 * bar_width / 2, adv_pred_probs[idx], bar_width, color='purple', label='Perturbed Image')

        # Highlighting specific columns
        orig_bars[target_pred].set_color('g')  # Set color of original image bar
        adv_bars[adv_pred].set_color('r')  # Set color of perturbed image bar

        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Prediction Probabilities for Noise Range {noise}')
        plt.xticks(np.arange(10))
        plt.legend()
        plt.grid(True)
        plt.show()






def base_test(model, device, test_loader, epsilon, attack):
    correct = 0
    correct_maj = 0
    diff_maj = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        if attack == "fgsm":
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack == "ifgsm":
            perturbed_data = ifgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        output_majority = predict(model, perturbed_data, num_of_samples=15, noise_range=110)

        final_pred = output.max(1, keepdim=True)[1]
        final_pred_maj = output_majority
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # experiment(model, data, perturbed_data, final_pred, target.item())
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        if final_pred_maj.item() == target.item():
            correct_maj += 1
        # else:
        #     test = predict(model, perturbed_data, visualize=True)

        if final_pred_maj.item() != final_pred.item():
            diff_maj += 1

    final_acc = correct / float(len(test_loader))
    final_acc_maj = correct_maj / float(len(test_loader))
    print(
        "Base Model - Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print("Majority Voting - Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct_maj, len(test_loader),
                                                                               final_acc_maj))

    return final_acc, final_acc_maj, adv_examples


def attacks(model):
    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    for attack in ("fgsm", "ifgsm"):
        # for attack in ("fgsm"):
        accuracies = []
        accuracies_maj_def = []
        examples = []
        for eps in epsilons:
            acc, acc_maj_def, ex = base_test(model, device, test_loader, eps, attack)
            accuracies.append(acc)
            accuracies_maj_def.append(acc_maj_def)
            examples.append(ex)
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, "*-", label="Without Majority Voting Defense")
        plt.plot(epsilons, accuracies_maj_def, "o-", label="With Majority Voting Defense")
        plt.title('Base Model: ' + attack)
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

        # cnt = 0
        # plt.figure(figsize=(8, 10))
        # for i in range(len(epsilons)):
        #     for j in range(len(examples[i])):
        #         cnt += 1
        #         plt.subplot(len(epsilons), len(examples[0]), cnt)
        #         plt.xticks([], [])
        #         plt.yticks([], [])
        #         if j == 0:
        #             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        #         orig, adv, ex = examples[i][j]
        #         plt.title("{} -> {}".format(orig, adv))
        #         plt.imshow(ex, cmap="gray")
        # plt.tight_layout()
        # plt.show()


class NetF(nn.Module):
    def __init__(self):
        super(NetF, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class NetF1(nn.Module):
    def __init__(self):
        super(NetF1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def defense(device, train_loader, val_loader, test_loader, epochs, Temp, epsilons):
    if not os.path.exists(F_MODEL_PATH):
        modelF = NetF().to(device)
        optimizerF = optim.Adam(modelF.parameters(), lr=0.0001, betas=(0.9, 0.999))
        schedulerF = optim.lr_scheduler.ReduceLROnPlateau(optimizerF, mode='min', factor=0.1, patience=3)
        criterion = nn.NLLLoss()
        model, lossF, val_lossF = defense_fit(modelF, device, optimizerF, schedulerF, criterion, train_loader,
                                              val_loader, Temp,
                                              epochs)
        torch.save(model, F_MODEL_PATH)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(np.arange(1, epochs + 1), lossF, "*-", label="Loss")
        plt.plot(np.arange(1, epochs + 1), val_lossF, "o-", label="Val Loss")
        plt.title("Network F")
        plt.xlabel("Num of epochs")
        plt.legend()
        plt.show()
    else:
        modelF = torch.load(F_MODEL_PATH, map_location=torch.device('cpu'))

    modelF.eval()

    # converting target labels to soft labels
    for data in train_loader:
        input, label = data[0].to(device), data[1].to(device)
        softlabel = F.log_softmax(modelF(input), dim=1)
        data[1] = softlabel

    if not os.path.exists(F1_MODEL_PATH):
        modelF1 = NetF1().to(device)
        optimizerF1 = optim.Adam(modelF1.parameters(), lr=0.0001, betas=(0.9, 0.999))
        schedulerF1 = optim.lr_scheduler.ReduceLROnPlateau(optimizerF1, mode='min', factor=0.1, patience=3)
        criterion = nn.NLLLoss()
        model, lossF1, val_lossF1 = defense_fit(modelF1, device, optimizerF1, schedulerF1, criterion, train_loader,
                                                val_loader,
                                                Temp,
                                                epochs)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(np.arange(1, epochs + 1), lossF1, "*-", label="Loss")
        plt.plot(np.arange(1, epochs + 1), val_lossF1, "o-", label="Val Loss")
        plt.title("Network F'")
        plt.xlabel("Num of epochs")
        plt.legend()
        plt.show()

        torch.save(model, F1_MODEL_PATH)
    else:
        modelF1 = torch.load(F1_MODEL_PATH, map_location=torch.device('cpu'))

    model = NetF1().to(device)
    model.load_state_dict(modelF1.state_dict())
    model.eval()

    for attack in ("fgsm", "ifgsm"):
        accuracies = []
        accuracies_maj_def = []
        examples = []
        for eps in epsilons:
            acc, acc_maj_def, ex = defense_test(model, device, test_loader, eps, 1, attack)
            accuracies.append(acc)
            accuracies_maj_def.append(acc_maj_def)
            examples.append(ex)
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, "*-", label="Without Majority Voting Defense")
        plt.plot(epsilons, accuracies_maj_def, "o-", label="With Majority Voting Defense")
        plt.title('Adversarial Trained Model: ' + attack)
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

        # cnt = 0
        # plt.figure(figsize=(8, 10))
        # for i in range(len(epsilons)):
        #     for j in range(len(examples[i])):
        #         cnt += 1
        #         plt.subplot(len(epsilons), len(examples[0]), cnt)
        #         plt.xticks([], [])
        #         plt.yticks([], [])
        #         if j == 0:
        #             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        #         orig, adv, ex = examples[i][j]
        #         plt.title("{} -> {}".format(orig, adv))
        #         plt.imshow(ex, cmap="gray")
        # plt.tight_layout()
        # plt.show()


def main():
    if not os.path.exists(BASE_MODEL_PATH):
        model = Net().to(device)
        model, loss, val_loss = base_fit(model, device, train_loader, val_loader, 10)
        training_results(loss, val_loss)
        torch.save(model, BASE_MODEL_PATH)
    else:
        model = torch.load(BASE_MODEL_PATH, map_location=torch.device('cpu'))

    model.eval()
    attacks(model)
    temp = 100
    epochs = 10
    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    # epsilons = [0.5]
    defense(device, train_loader, val_loader, test_loader, epochs, temp, epsilons)


def fgsm_attack(input, epsilon, data_grad):
    pert_out = input + epsilon * data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    return pert_out


def ifgsm_attack(input, epsilon, data_grad):
    iter = 10
    alpha = epsilon / iter
    pert_out = input
    for i in range(iter - 1):
        pert_out = pert_out + alpha * data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)
        if torch.norm((pert_out - input), p=float('inf')) > epsilon:
            break
    return pert_out


def defense_fit(model, device, optimizer, scheduler, criterion, train_loader, val_loader, Temp, epochs):
    data_loader = {'train': train_loader, 'val': val_loader}
    print("Fitting the model...")
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        loss_per_epoch, val_loss_per_epoch = 0, 0
        for phase in ('train', 'val'):
            for i, data in enumerate(data_loader[phase]):
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                output = F.log_softmax(output / Temp, dim=1)
                # calculating loss on the output
                loss = criterion(output, label)
                if phase == 'train':
                    optimizer.zero_grad()
                    # grad calc w.r.t Loss func
                    loss.backward()
                    # update weights
                    optimizer.step()
                    loss_per_epoch += loss.item()
                else:
                    val_loss_per_epoch += loss.item()
        scheduler.step(val_loss_per_epoch / len(val_loader))
        print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch + 1, loss_per_epoch / len(train_loader),
                                                       val_loss_per_epoch / len(val_loader)))
        train_loss.append(loss_per_epoch / len(train_loader))
        val_loss.append(val_loss_per_epoch / len(val_loader))
    return model, train_loss, val_loss


def defense_test(model, device, test_loader, epsilon, Temp, attack):
    correct = 0
    correct_maj = 0
    diff_maj = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        output = F.log_softmax(output / Temp, dim=1)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        if attack == "fgsm":
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack == "ifgsm":
            perturbed_data = ifgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        output_majority = predict(model, perturbed_data, num_of_samples=100, noise_range=110)

        final_pred = output.max(1, keepdim=True)[1]
        final_pred_maj = output_majority

        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # experiment(model, data, perturbed_data, final_pred, target.item())
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        if final_pred_maj.item() == target.item():
            correct_maj += 1
        # else:
        #     test = predict(model, perturbed_data, visualize=True)

        if final_pred_maj.item() != final_pred.item():
            diff_maj += 1

    final_acc = correct / float(len(test_loader))
    final_acc_maj = correct_maj / float(len(test_loader))
    print("Base Defensive Model - Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader),
                                                                                    final_acc))
    print(
        "Defensive Model with Majority Voting - Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct_maj,
                                                                                                  len(test_loader),
                                                                                                  final_acc_maj))

    return final_acc, final_acc_maj, adv_examples


if __name__ == "__main__":
    main()
