#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np


###############################################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 512

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



class Net(nn.Module):
    def __init__(self, p_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class Net2(nn.Module):
    def __init__(self, p_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 10)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x




def train_with_dropout(p_dropout, num_epochs=1, learning_rate=0.01):

    net = Net(p_dropout=p_dropout)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        net.train(True)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

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


    n_train = trainloader.dataset.data.shape[0]
    missclassified_train = 0
    net.train(False)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        missclassified_train += torch.sum(predicted != labels).item()


    n_test = testloader.dataset.data.shape[0]
    missclassified_test = 0
    net.train(False)
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        missclassified_test += torch.sum(predicted != labels).item()

    classification_error_train = missclassified_train/n_train*100
    classification_error_test = missclassified_test/n_test*100

    return net, losses, classification_error_train, classification_error_test




dropout_values = [0.2, 0.4, 0.5, 0.6, 0.8]
num_epochs = 15
learning_rate = 0.01
res_net = []
res_losses = []
res_train = []
res_test = []
for p_dropout in dropout_values:
    net, losses, c_error_train, c_error_test = train_with_dropout(p_dropout, num_epochs=num_epochs, learning_rate=learning_rate)
    res_net.append(net)
    res_losses.append(losses)
    res_train.append(c_error_train)
    res_test.append(c_error_test)




fig, ax = plt.subplots()
ax.plot(dropout_values, res_train, 'bo-', label='train')
ax.plot(dropout_values, res_test, 'ro-', label='test')
ax.legend()
ax.set_xlabel('p_dropout', fontsize=14)
ax.set_ylabel('classification error', fontsize=14)
fig.savefig('/tmp/classification_error.pdf', bbox_inches='tight')

fig, ax = plt.subplots()
for i, p_dropout in enumerate(dropout_values):
    ax.plot(res_losses[i], ls='-', label=str(p_dropout))
ax.legend()
ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)
fig.savefig('/tmp/loss.pdf', bbox_inches='tight')



###############################################################################


dataiter = iter(trainloader)
images, labels = next(dataiter)

x = images[0].reshape(1,*images[0].shape).cuda()

res_net[1].train(True)
res_net[1].forward(x)

res_net[1].train(False)
res_net[1].forward(x)

###############################################################################



#net, losses, c_error_train, c_error_test = train_with_dropout(0.5, num_epochs=15, learning_rate=learning_rate)
net = res_net[1]

# untrained network
net = Net()


conv1weights = net.conv1.weight.data.cpu().numpy()
fig, ax = plt.subplots(conv1weights.shape[0], 1, figsize=(5, 5))
for i in range(conv1weights.shape[0]):
    ax[i].imshow(conv1weights[i,0,:,:], cmap='gray_r')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

conv2weights = net.conv2.weight.data.cpu().numpy()
fig, ax = plt.subplots(conv2weights.shape[0], conv2weights.shape[1], figsize=(10, 10))
for i in range(conv2weights.shape[0]):
    for j in range(conv2weights.shape[1]):
        ax[i,j].imshow(conv2weights[i,j,:,:], cmap='gray_r')
        ax[i,j].set_axis_off()
fig.tight_layout()
plt.show()


###############################################################################

net = res_net[1]

dataiter = iter(trainloader)
images, labels = next(dataiter)

x = images[0].reshape(1,*images[0].shape).cuda()
x.shape
x = net.conv1(x)
x.shape
x = F.relu(x)
x.shape
x = net.pool(x)
x.shape
x = net.conv2(x)
x.shape
x = F.relu(x)
x.shape
x = net.pool(x)
x.shape
x = torch.flatten(x, 1)
x.shape
x = net.fc1(x)
x.shape
x = F.relu(x)
x.shape
x = net.fc2(x)
x.shape
m = nn.Softmax(dim=1)
m(x).shape
m(x)


plt.imshow(images[0].reshape(28,28), cmap='gray_r')
plt.show()




###############################################################################



class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.linear(x)
        return x




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 512

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
subset_indices = ((trainset.targets == 1) + (trainset.targets == 7)).nonzero().view(-1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices))

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
subset_indices = ((testset.targets == 1) + (testset.targets == 7)).nonzero().view(-1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False,
                                         sampler=SubsetRandomSampler(subset_indices))


dim = torch.prod(torch.as_tensor(trainset.data[0].shape)).item()
logreg = LogisticRegression(dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logreg.to(device)



num_epochs = 20
learning_rate = 0.1
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(logreg.parameters(), lr=learning_rate)

losses = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    logreg.train(True)
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        labels[labels==7] = 0
        labels = labels.reshape(-1,1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = logreg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += outputs.shape[0] * loss.item()

    # print and save loss after epoch
    print(f'epoch: {epoch} loss: {epoch_loss / len(trainset)}')
    losses.append(epoch_loss / len(trainset))



n_train = trainloader.dataset.data.shape[0]
missclassified_train = 0
logreg.train(False)
for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    labels[labels==7] = 0
    labels = labels.reshape(-1,1)
    outputs = logreg(inputs)
    predicted = outputs > 0
    missclassified_train += torch.sum(predicted != labels).item()


n_test = testloader.dataset.data.shape[0]
missclassified_test = 0
logreg.train(False)
for i, data in enumerate(testloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    labels[labels==7] = 0
    labels = labels.reshape(-1,1)
    outputs = logreg(inputs)
    predicted = outputs > 0
    missclassified_test += torch.sum(predicted != labels).item()

classification_error_train = missclassified_train/n_train*100
classification_error_test = missclassified_test/n_test*100

###############################################################################

plt.imshow(logreg.linear.weight.data.cpu().numpy().reshape(28,28))
plt.show()


###############################################################################


def train_logreg_with_regularization(lambda1=0.0, lambda2=0.0):
    dim = torch.prod(torch.as_tensor(trainset.data[0].shape)).item()
    logreg = LogisticRegression(dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logreg.to(device)

    num_epochs = 20
    learning_rate = 0.01
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(logreg.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        logreg.train(True)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            labels[labels==7] = 0
            labels = labels.reshape(-1,1).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # regularizer
            l1_reg = lambda1 * torch.linalg.norm(logreg.linear.weight.view(-1), 1)
            l2_reg = lambda2 * torch.pow(torch.linalg.norm(logreg.linear.weight.view(-1), 2), 2)

            # forward + backward + optimize
            outputs = logreg(inputs)
            loss = criterion(outputs, labels)
            loss += l1_reg
            loss += l2_reg
            loss.backward()
            optimizer.step()

            epoch_loss += outputs.shape[0] * loss.item()

        # print and save loss after epoch
        print(f'epoch: {epoch} loss: {epoch_loss / len(trainset)}')
        losses.append(epoch_loss / len(trainset))


    n_train = trainloader.dataset.data.shape[0]
    missclassified_train = 0
    logreg.train(False)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels[labels==7] = 0
        labels = labels.reshape(-1,1)
        outputs = logreg(inputs)
        predicted = outputs > 0
        missclassified_train += torch.sum(predicted != labels).item()


    n_test = testloader.dataset.data.shape[0]
    missclassified_test = 0
    logreg.train(False)
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels[labels==7] = 0
        labels = labels.reshape(-1,1)
        outputs = logreg(inputs)
        predicted = outputs > 0
        missclassified_test += torch.sum(predicted != labels).item()

    classification_error_train = missclassified_train/n_train*100
    classification_error_test = missclassified_test/n_test*100

    return logreg, losses, classification_error_train, classification_error_test


lambda_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
res_logreg = []
res_losses = []
res_train = []
res_test = []
for l in lambda_values:
    logreg, losses, classification_error_train, classification_error_test = train_logreg_with_regularization(lambda2=l)
    res_logreg.append(logreg)
    res_losses.append(losses)
    res_train.append(classification_error_train)
    res_test.append(classification_error_test)



fig, ax = plt.subplots()
ax.plot(lambda_values, res_train, 'bo-', label='train')
ax.plot(lambda_values, res_test, 'ro-', label='test')
ax.legend()
ax.set_xlabel('lambda', fontsize=14)
ax.set_ylabel('classification error', fontsize=14)
fig.savefig('/tmp/logreg_classification_error.pdf', bbox_inches='tight')

fig, ax = plt.subplots()
for i, lambda_value in enumerate(lambda_values):
    ax.plot(res_losses[i], ls='-', label=str(lambda_value))
ax.legend()
ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)
fig.savefig('/tmp/logreg_loss.pdf', bbox_inches='tight')