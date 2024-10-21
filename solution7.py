import torch
import torchvision
import torchvision.transforms as T

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

##################################
# Task 17: Denoising Autoencoder
##################################

class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x + torch.randn(x.shape) * self.std + self.mean
        return x

    def __repr__(self):
        return '{}(mean={}, std={})'.format(
            self.__class__.__name__,
            self.mean,
            self.std)


transform = T.ToTensor()
corrupt_noise = GaussianNoise(0.0, 0.1)

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform)

loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=600,
    shuffle=True,
    num_workers=0)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


model = AutoEncoder(28 * 28)
criterion = nn.MSELoss()
num_epochs = 5
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2)

losses = []
for epoch in range(num_epochs):
    # model.train(True)
    epoch_loss = 0.0
    for inputs, _ in loader:
        optimizer.zero_grad()
        inputs = inputs.view(inputs.shape[0], -1)
        corrupted_inputs = corrupt_noise(inputs)
        # outputs = model(corrupted_inputs)
        z = model.encoder(corrupted_inputs)
        outputs = model.decoder(z)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(loader)
    print(f'epoch: {epoch} loss={epoch_loss:.4f}')
    losses.append(epoch_loss)


inputs, labels = next(iter(loader))
inputs = inputs.view(inputs.shape[0], -1)
corrupted_inputs = corrupt_noise(inputs)
model.train(False)
z = model.encoder(corrupted_inputs)
x_rec = model.decoder(z)

# Compare inputs/outputs
fig, ax = plt.subplots(
    nrows=10,
    ncols=3,
    figsize=(8, 9 * 3))
for i in range(10):
    # original
    img = inputs[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i, 0].imshow(img)

    # corrupted
    img = corrupted_inputs[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i, 1].imshow(img)

    # reconstruction
    img = x_rec[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i, 2].imshow(img)
plt.show()


# Shows latent space
fig, ax = plt.subplots(figsize=(10, 10))
xy = z.cpu().detach().numpy()
ax.scatter(xy[:, 0], xy[:, 1],
           c=labels.cpu().detach().numpy(),
           cmap='tab10',
           alpha=0.5)
plt.show()


# Pick two points in latent space and interpolate
num_points = 7
alphas = torch.linspace(0, 1, num_points)
z_interpolation = torch.stack([
    z[5] * alpha + z[1] * (1.0 - alpha)
    for alpha in alphas
], dim=0)
interpolation_rec = model.decoder(z_interpolation)

fig, ax = plt.subplots(
    nrows=1,
    ncols=num_points,
    figsize=(2 * num_points, 6))
for i in range(num_points):
    img = interpolation_rec[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i].imshow(img)
plt.show()


# Shows latent space
fig, ax = plt.subplots(figsize=(10, 10))
xy = z.cpu().detach().numpy()
ax.scatter(xy[:, 0], xy[:, 1],
           c=labels.cpu().detach().numpy(),
           cmap='tab10',
           alpha=0.5)
xy_line = z_interpolation.cpu().detach().numpy()
ax.plot(xy_line[:, 0], xy_line[:, 1], 'r-')
ax.plot(xy_line[:, 0], xy_line[:, 1], 'ro', alpha=0.1)
plt.show()


##################################
# Task 18: A first generative model
##################################

# z = model.encoder(inputs)
# Make z ourselves
mu = torch.ones(2)
std = torch.eye(2) * 5.0
dist = torch.distributions.MultivariateNormal(
    mu, std)
z_sample = dist.sample((10,))
x_rec = model.decoder(z_sample)


# Shows latent space
fig, ax = plt.subplots(figsize=(10, 10))
xy = z.cpu().detach().numpy()
ax.scatter(xy[:, 0], xy[:, 1],
           c=labels.cpu().detach().numpy(),
           cmap='tab10',
           alpha=0.5)
xy_samples = z_sample.cpu().detach().numpy()
ax.plot(xy_samples[:, 0], xy_samples[:, 1],
        'r+', ms=20)
plt.show()


fig, ax = plt.subplots(
    nrows=10,
    ncols=1,
    figsize=(8, 9))
for i in range(10):
    # reconstruction
    img = x_rec[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i].imshow(img)
plt.show()


##################################
# Task 19: Variational AutoEncoder
##################################

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h[:, 0::2], h[:, 1::2]
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.training:
            eps = torch.randn_like(mu)
            var = torch.exp(0.5 * log_var)
            z = mu + eps * var
        else:
            z = mu
        x_rec = self.decode(z)
        return x_rec, mu, log_var


def kl_div(mu, log_var):
    var = torch.exp(0.5 * log_var)
    kld = 0.5 * torch.sum(var + mu * mu - 1 - log_var, dim=-1)
    kld = torch.mean(kld, dim=0)
    return kld


model = VariationalAutoEncoder(28 * 28)
criterion = nn.MSELoss(reduction='sum')
num_epochs = 10
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2)

losses = []
for epoch in range(num_epochs):
    # model.train(True)
    epoch_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        inputs = inputs.view(inputs.shape[0], -1)
        # corrupted_inputs = corrupt_noise(inputs)
        # outputs = model(corrupted_inputs)
        outputs, mu, log_var = model(inputs)
        loss_rec = torch.sum(torch.square(outputs - inputs))
        loss_rec = loss_rec / inputs.shape[0]
        loss_kld = kl_div(mu, log_var)
        loss = loss_rec + loss_kld
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(f'loss_rec={loss_rec.item()} loss_kld={loss_kld.item()}')
    epoch_loss = epoch_loss / len(loader)
    print(f'epoch: {epoch} loss={epoch_loss:.4f}')
    losses.append(epoch_loss)


inputs, labels = next(iter(loader))
inputs = inputs.view(inputs.shape[0], -1)
model.train(False)
mu, log_var = model.encode(inputs)
x_rec = model.decode(mu)

# Compare inputs/outputs
fig, ax = plt.subplots(
    nrows=10,
    ncols=2,
    figsize=(8, 9))
for i in range(10):
    # original
    img = inputs[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i, 0].axis('off')
    ax[i, 0].imshow(img)

    # reconstruction
    img = x_rec[i].cpu().detach().numpy()
    img = img.reshape(28, 28)
    ax[i, 1].axis('off')
    ax[i, 1].imshow(img)
plt.show()


# Shows latent space
fig, ax = plt.subplots(figsize=(10, 10))
xy = mu.cpu().detach().numpy()
ax.scatter(xy[:, 0], xy[:, 1],
           c=labels.cpu().detach().numpy(),
           cmap='tab10',
           alpha=0.5)
plt.show()
