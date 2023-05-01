# DCGAN
This repository contains code for training a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch. 
The implementation of the DCGAN was carried out with the primary objective of gaining a deeper understanding of GAN training. 
It was trained on the [Pokémon dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset).

<p align="center">
  <img src="pokemon_animation.gif" alt="animated" />
</p>

The model will be trained with the following hyperparameters, which are specified in 'config.yaml':
* num_epochs: 1000
* batch_size: 128
* image_size: 64
* latent_dim: 100
* ngf: 64
* ndf: 64
* nc: 3
* lr: 0.0002
* beta1: 0.5
* beta2: 0.999
