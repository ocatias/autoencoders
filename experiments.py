import autoencoder.autoencoder as  autoencoder
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import cv2
import numpy as np
num_epochs = 15


def main():
    model = autoencoder.Autoencoder()

    batch_size = 50
    t = transforms.Compose([
                       transforms.ToTensor()]
                       )

    data_train = DataLoader( torchvision.datasets.MNIST('/data/MNIST', download=True, train=True, transform=t),
        batch_size=batch_size, drop_last=True, shuffle=True)
    data_valid = DataLoader( torchvision.datasets.MNIST('/data/mnist', download=True, train=False, transform=t),
        batch_size=batch_size, drop_last=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    for curr_epoch in range(num_epochs):
        epoch_loss = 0
        store_loss_x = 0
        store_loss_dkl = 0
        for (data, _) in data_train:
            optimizer.zero_grad()
            data_shaped = data.view(data.shape[0], -1)

            x_hat, KL = model(data_shaped)
            loss = nn.functional.binary_cross_entropy(x_hat, data_shaped, reduction='sum')

            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()

        print(curr_epoch, ": ", epoch_loss)

    for (data, _) in data_train:
        data_shaped = data.view(data.shape[0], -1)
        x, _ = model(data_shaped)
        optimizer.step()
        cv2.imshow("win", data_shaped[0].view(28,28).numpy())
        cv2.waitKey(0)
        cv2.imshow("win", x[0].view(28,28).detach().numpy())
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
