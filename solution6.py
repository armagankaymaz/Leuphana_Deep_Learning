#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import sklearn.datasets

import matplotlib.pyplot as plt
import numpy as np



# preliminary solution

###############################################################################

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
])

batch_size = 600

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
dataiter = iter(trainloader)
images, labels = dataiter.next()

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32,2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Tanh(),
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


net = AutoEncoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

learning_rate = 0.01
num_epochs = 20

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

losses = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    net.train(True)
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.size(0), -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        epoch_loss += outputs.shape[0] * loss.item()

    # print and save loss after epoch
    print(f'epoch: {epoch} loss: {epoch_loss / len(trainset)}')
    losses.append(epoch_loss / len(trainset))




net.train(False)
z = net.encoder(inputs)
x_rec = net.decoder(z)

fig, ax = plt.subplots(10,2, figsize=(8,18))
for i in range(10):
    # original
    img = inputs[i].cpu().detach().numpy() / 2 + 0.5
    img = img.reshape(28,28)
    ax[i,0].imshow(img)
    # reconstruction
    img = x_rec[i].cpu().detach().numpy() / 2 + 0.5
    img = img.reshape(28,28)
    ax[i,1].imshow(img)



fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(z[:,0].cpu().detach().numpy(),
           z[:,1].cpu().detach().numpy(),
           c=labels.cpu().detach().numpy())


alpha = np.linspace(0,1,7)
z_start = net.encoder(inputs[0]).cpu().detach().numpy()
z_end = net.encoder(inputs[5]).cpu().detach().numpy()


z_interpolation = torch.tensor([(1-a)*z_start+a*z_end for a in alpha]).to(device)
interpolation_rec = net.decoder(z_interpolation)



fig, ax = plt.subplots(1,7, figsize=(14,6))
for i in range(7):
    # reconstruction
    img = interpolation_rec[i].cpu().detach().numpy() / 2 + 0.5
    img = img.reshape(28,28)
    ax[i].imshow(img)











