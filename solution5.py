#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import sklearn.datasets

import matplotlib.pyplot as plt
import numpy as np


###############################################################################

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
     transforms.RandomCrop((20,20))
])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          #shuffle=True,
                                          num_workers=2,
                                          sampler=SubsetRandomSampler([7]))
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images[0].shape)

fig, ax = plt.subplots(figsize=(2,2))
img = images[0].numpy() / 2 + 0.5
ax.imshow(np.transpose(img, (1, 2, 0)))
plt.show()


###############################################################################

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform = transforms.Compose([
     transforms.ToTensor(),
     #transforms.Normalize((0.5), (0.5)),
     AddGaussianNoise(0., 0.05)
])

batch_size = 1

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          #shuffle=True,
                                          num_workers=0,
                                          sampler=SubsetRandomSampler([8]))
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images[0].shape)

fig, ax = plt.subplots(figsize=(2,2))
img = images[0].numpy() / 2 + 0.5
ax.imshow(np.transpose(img, (1, 2, 0)))
plt.show()
###############################################################################

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
     transforms.RandomVerticalFlip(p=0.5)
])

batch_size = 1

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          #shuffle=True,
                                          num_workers=2,
                                          sampler=SubsetRandomSampler([1]))
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images[0].shape)

fig, ax = plt.subplots(figsize=(2,2))
img = images[0].numpy() / 2 + 0.5
ax.imshow(np.transpose(img, (1, 2, 0)))
plt.show()
###############################################################################

x, y = sklearn.datasets.make_moons(n_samples=100, shuffle=True,
                                   noise=0.05, random_state=0)
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(x[y==1,0], x[y==1,1], 'ro', label="positive class")
ax.plot(x[y==0,0], x[y==0,1], 'bo', label="negative class")
ax.legend()
plt.show()

x = torch.as_tensor(x, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(x, y)

batch_size = 16

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

dataiter = iter(trainloader)
points, labels = next(dataiter)

fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(x[y==1,0], x[y==1,1], 'ro', alpha=0.1)
ax.plot(x[y==0,0], x[y==0,1], 'bo', alpha=0.1)
ax.plot(points[labels==1,0], points[labels==1,1], 'ro')
ax.plot(points[labels==0,0], points[labels==0,1], 'bo')
plt.show()

###############################################################################

class TwoMoonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x



net = TwoMoonNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

num_epochs = 50
learning_rate = 0.1
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

losses = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    net.train(True)
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.reshape(-1,1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += outputs.shape[0] * loss.item()

    # print and save loss after epoch
    print(f'epoch: {epoch} loss: {epoch_loss / len(trainset)}')
    losses.append(epoch_loss / len(trainset))



fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(x[y==1,0], x[y==1,1], 'ro', label="positive class")
ax.plot(x[y==0,0], x[y==0,1], 'bo', label="negative class")
ax.legend()
x_min = x[:, 0].min().item()-1
x_max = x[:, 0].max().item()+1
y_min = x[:, 1].min().item()-1
y_max = x[:, 1].max().item()+1
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
XY = np.vstack((XX.ravel(), YY.ravel())).T
# compute forward pass on every grid point
Z = F.sigmoid(net(torch.as_tensor(XY, dtype=torch.float).to(device))).reshape(XX.shape).cpu().detach().numpy()
# show prediction area
ax.pcolormesh(XX, YY, Z, cmap=plt.cm.coolwarm)
# show separating line
ax.contour(XX, YY, Z, colors=['k'], linestyles=['-'], levels=[.5])
plt.show()
fig.savefig('/tmp/ex8_twomoons.pdf', bbox_inches='tight')





###############################################################################




class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 10)
        self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = nn.ReLU()(self.linear3(x))
        x = nn.ReLU()(self.linear4(x))
        x = self.linear5(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)


class Net_bn_before(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(input_dim, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
        self._initialize_weights()


    def forward(self, x):
        x = self.model(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)
    
    


class Net_bn_after(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.BatchNorm1d(256),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.BatchNorm1d(64),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.BatchNorm1d(32),
                        nn.Linear(32, 10)
                    )
        self._initialize_weights()


    def forward(self, x):
        x = self.model(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

# Define function to train and evaluate the model
def train_and_evaluate(net, trainloader, testloader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_losses = []
    test_accuracies = []
    num_layers = len(list(filter(lambda x: isinstance(x, nn.Linear), net.modules())))
    grad_norms = np.zeros((num_epochs, num_layers, len(trainloader)))
    
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            for j, layer in enumerate(filter(lambda x: isinstance(x, nn.Linear), net.modules())):
                grad_norms[epoch, j, i] = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in layer.parameters()]))
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(trainloader))

        # Evaluate on test set
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies, grad_norms



transform1 = transforms.Compose([
     transforms.ToTensor(),
])

transform2 = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
])

transform3 = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((-100.0), (1.0)),
])


batch_size = 128
learning_rate = 0.01
num_epochs = 10

datasets = {
    "MNIST": torchvision.datasets.MNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "CIFAR10": torchvision.datasets.CIFAR10
}

transforms = [transform1, transform2, transform3]
results = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train and evaluate for each dataset, transformation, and network type
for dataset_name, dataset in datasets.items():
    for i, transform in enumerate(transforms):
        trainset = dataset(root='./data', train=True, download=True, transform=transform)
        testset = dataset(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        input_dim = torch.prod(torch.tensor(trainset.data[0].shape)).item()

        for net_name, net_class in [("Net", Net), ("Net_bn_before", Net_bn_before), ("Net_bn_after", Net_bn_after)]:
            print(f"Training {net_name} on {dataset_name} with transform {i + 1}")
            net = net_class(input_dim).to(device)
            train_losses, test_accuracies, grad_norms = train_and_evaluate(net, trainloader, testloader, num_epochs, learning_rate, device)
            results[(dataset_name, i + 1, net_name)] = (train_losses, test_accuracies, grad_norms)

# Visualize results
fig, axs = plt.subplots(len(datasets), 3, figsize=(15, 5 * len(datasets)))
for dataset_idx, (dataset_name, dataset) in enumerate(datasets.items()):
    for transform_idx in range(3):
        for net_idx, net_name in enumerate(["Net", "Net_bn_before", "Net_bn_after"]):
            key = (dataset_name, transform_idx + 1, net_name)
            if key in results:
                train_losses, test_accuracies, grad_norms = results[key]
                ax = axs[dataset_idx, transform_idx]
                ax.plot(test_accuracies, label=f"{net_name}")
                ax.set_title(f"{dataset_name} - Transform {transform_idx + 1}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.legend()

plt.tight_layout()
plt.show()

import os
os.makedirs("./results", exist_ok=True)

# Save results to files
for (dataset_name, transform_idx, net_name), (train_losses, test_accuracies, grad_norms) in results.items():
    np.save(f"./results/{dataset_name}_transform{transform_idx}_{net_name}_train_losses.npy", train_losses)
    np.save(f"./results/{dataset_name}_transform{transform_idx}_{net_name}_test_accuracies.npy", test_accuracies)
    np.save(f"./results/{dataset_name}_transform{transform_idx}_{net_name}_grad_norms.npy", grad_norms)

print("Training and evaluation completed. Results saved to './results/' directory.")