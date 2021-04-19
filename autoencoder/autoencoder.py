import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dims)
        self.decoder = Decoder(input_dim, latent_dims)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        kl = 0.5*torch.sum(torch.exp(log_var) + torch.square(mean) - 1 - log_var)
        return self.decoder(z), kl

class Encoder(nn.Module):
    """
    Encodes a picture in output_dim dimensions.
    """
    def __init__(self, input_dim = 784, output_dim = 2, layer_1 = 500, layer_2 = 100):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, output_dim*2)

    def forward(self, x):
        mean, log_var = torch.chunk(self.fc3(self.fc2(torch.relu(self.fc1(x)))), 2, -1)
        z = sample(mean, log_var)
        return z, mean, log_var



def sample(mean, log_variance):
    dimensions = mean.size()
    # toch.randn draws from a normal distribution!
    eps = torch.rand_like(mean)
    z = mean + torch.exp(log_variance / 2) * eps
    return z


class Decoder(nn.Module):
    def __init__(self, output_dim = 784, input_dim = 2, layer_1 = 100, layer_2 = 500):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigm(self.fc3(x))
