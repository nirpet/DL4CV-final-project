import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from inference import predict

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", message="dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release.")


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
BASE_MODEL_PATH = 'models/base_model.pth'
F_MODEL_PATH = 'models/f_model.pth'
F1_MODEL_PATH = 'models/f1_model.pth'

# np.random.seed(42)
# torch.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
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


def base_test(model, device, test_loader, epsilon, attack, with_majority_defense=False):
    correct = 0
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
        elif attack == "mifgsm":
            perturbed_data = mifgsm_attack(data, epsilon, data_grad)

        if with_majority_defense:
            output = predict(model, perturbed_data)
        else:
            output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


def attacks(model):
    epsilons = [0, 0.1, 0.2, 0.3]
    for attack in ("fgsm", "ifgsm", "mifgsm"):
        accuracies = []
        accuracies_maj_def = []
        examples = []
        for eps in epsilons:
            acc, ex = base_test(model, device, test_loader, eps, attack)
            accuracies.append(acc)
            examples.append(ex)
            acc_maj_def, _ = base_test(model, device, test_loader, eps, attack, True)
            accuracies_maj_def.append(acc_maj_def)
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, "*-", label="Without Majority Voting Defense")
        plt.plot(epsilons, accuracies_maj_def, "o-", label="With Majority Voting Defense")
        plt.title(attack)
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
        model, lossF, val_lossF = defense_fit(modelF, device, optimizerF, schedulerF, criterion, train_loader, val_loader, Temp,
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
        model, lossF1, val_lossF1 = defense_fit(modelF1, device, optimizerF1, schedulerF1, criterion, train_loader, val_loader,
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
    for attack in ("fgsm", "ifgsm", "mifgsm"):
        accuracies = []
        accuracies_maj_def = []
        examples = []
        for eps in epsilons:
            acc, ex = defense_test(model, device, test_loader, eps, 1, attack)
            accuracies.append(acc)
            examples.append(ex)
            acc_maj_def, _ = defense_test(model, device, test_loader, eps, 1, attack=attack, with_majority_voting=True)
            accuracies_maj_def.append(acc_maj_def)
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, "*-", label="Without Majority Voting Defense")
        plt.plot(epsilons, accuracies_maj_def, "o-", label="With Majority Voting Defense")
        plt.title(attack)
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
    attacks(model)
    temp = 100
    epochs = 10
    epsilons = [0, 0.1, 0.2, 0.3]
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


def mifgsm_attack(input, epsilon, data_grad):
    iter = 10
    decay_factor = 1.0
    pert_out = input
    alpha = epsilon / iter
    g = 0
    for i in range(iter - 1):
        g = decay_factor * g + data_grad / torch.norm(data_grad, p=1)
        pert_out = pert_out + alpha * torch.sign(g)
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


def defense_test(model, device, test_loader, epsilon, Temp, attack, with_majority_voting=False):
    correct = 0
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
        elif attack == "mifgsm":
            perturbed_data = mifgsm_attack(data, epsilon, data_grad)

        if with_majority_voting:
            output = predict(model, perturbed_data)
        else:
            output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


if __name__ == "__main__":
    main()
