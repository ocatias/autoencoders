import autoencoder.autoencoder as  autoencoder
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import cv2
import numpy as np
import os

num_epochs = 100

save_model_after_epoch = True

def main():
    device = "cuda"

    model = autoencoder.Autoencoder(3072, 2)
    model.to(device)
    batch_size = 1000
    t = transforms.Compose([
                       transforms.ToTensor()]
                       )

    #data_train = DataLoader( torchvision.datasets.MNIST('/data/MNIST', download=True, train=True, transform=t),
    #    batch_size=batch_size, drop_last=True, shuffle=True)
    #data_valid = DataLoader( torchvision.datasets.MNIST('/data/mnist', download=True, train=False, transform=t),
    #    batch_size=batch_size, drop_last=True, shuffle=True)

    print("Loading data")
    data_train = DataLoader( torchvision.datasets.CIFAR10('\data\CIFAR10', download=True, train=True, transform=t),
       batch_size=batch_size, drop_last=True, shuffle=True)
    data_valid = DataLoader( torchvision.datasets.CIFAR10('\data\cifar10', download=True, train=False, transform=t),
       batch_size=batch_size, drop_last=True, shuffle=True)

    print("Start training")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for curr_epoch in range(num_epochs):
        epoch_loss = 0
        store_loss_x = 0
        store_loss_dkl = 0
        for (data, _) in data_train:
            optimizer.zero_grad()
            data_shaped = data.view(data.shape[0], -1).to(device)

            x_hat, KL = model(data_shaped)
            loss = KL + nn.functional.binary_cross_entropy(x_hat, data_shaped, reduction='sum')

            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy()

        if save_model_after_epoch:
            torch.save(model.state_dict(), os.path.join("autoencoder", "models", "model_" + str(curr_epoch) + ".nn"))

        print(curr_epoch, ": ", epoch_loss / len(data_train))

    print("Start validation")
    model.eval()
    test_loss = 0
    for (data, _) in data_valid:
        data_shaped = data.view(data.shape[0], -1).to(device)
        x_hat, KL = model(data_shaped)
        loss = KL + nn.functional.binary_cross_entropy(x_hat, data_shaped, reduction='sum')
        test_loss += loss.cpu().detach().numpy()

        #cv2.imshow("win", data_shaped[0].cpu().view(28,28).numpy())
        #cv2.waitKey(0)
        #cv2.imshow("win", x[0].cpu().view(28,28).detach().numpy())
        #cv2.waitKey(0)

    print("Test loss: ", test_loss / len(data_valid))
    torch.save(model.state_dict(), os.path.join("autoencoder", "models", "model_final.nn"))

if __name__ == "__main__":
    main()
