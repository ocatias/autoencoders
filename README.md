# Variational Autoencoder
An implementation of a variational autoencoder (VAE), see [*Kingma and Welling (2013)*](https://arxiv.org/abs/1312.6114). Useful resources I used to understand VAEs:
- [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) by Jaan Altosaar
- [Variational Autoencoder: Intuition and Implementation](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/) by Augustinus Kristiadi

## Visualizations of the latent space
I embedded MNIST and Fashion MNIST in 2D space. The gifs are obtained by decoding points that lie on a circle in 2d space.

![](cycle_MNIST.gif) ![](cycle_FMNIST.gif)

<img src="latent_space_MNIST.png" alt="" width="400"> <img src="latent_space_FMNIST.png" alt="" width="400">
