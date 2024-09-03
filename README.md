# Wasserstein GAN with Weight Clipping

This repository contains the implementation of a Wasserstein Generative Adversarial Network (WGAN) with weight clipping, trained to generate human faces. The WGAN model improves upon traditional GANs by providing a more stable training process and better convergence properties by utilizing Wasserstein loss.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [License](#license)

## Overview

This project implements the Wasserstein GAN with weight clipping, as proposed in the paper "Wasserstein GAN" by Martin Arjovsky, Soumith Chintala, and Léon Bottou. The WGAN model addresses common issues in GAN training, such as mode collapse and unstable gradients, by minimizing the Wasserstein distance between the real and generated data distributions.

## Architecture

### Architecture Details

The WGAN model utilizes architecture as proposed in "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" paper and aims to stabilize the model training using wasserstein distance and weight clipping techniques as described in "Wasserstein GAN" paper: 

- **Generator**: Utilizes transposed convolutions to upsample the input noise vector into a desired output image size. The generator employs ReLU activation functions, except for the output layer which uses a Tanh activation function.
- **Discriminator (Critic)**: A convolutional neural network that distinguishes between real and fake samples. It uses Leaky ReLU activation functions. Unlike traditional GANs, the discriminator's output is not a probability but rather a real-valued score, which represents the Wasserstein distance.

### Key Differences from Standard GANs

- **Weight Clipping**: To enforce the Lipschitz constraint required by the Wasserstein distance, the weights of the discriminator are clipped to a fixed range after each update.
- **No Sigmoid in Discriminator**: The discriminator outputs a real number without applying a sigmoid function, allowing it to represent the Wasserstein distance.

## Results

The training process was monitored by generating samples from a batch of fixed noise at regular intervals. A timelapse of these generated samples is available, showcasing the model's progression over time. Images were captured every 100th batch during each epoch, allowing you to visualize how the quality and diversity of the generated data improve as the training progresses.

The timelapse demonstrates:

- **Early Epochs**: The model starts by generating solid-colored images very early in the training. This is a common behavior as the model begins to learn the basic structure of the data distribution.
- **Mid to Late Epochs**: The model begins to produce more coherent and realistic outputs, with finer details and improved structure.
- **Final Epochs**: The generated samples closely resemble the target data distribution, demonstrating the effectiveness of the WGAN architecture with weight clipping.

![Generated Image](output_F4IVQ1.gif)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
