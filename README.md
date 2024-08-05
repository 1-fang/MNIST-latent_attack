# Adversarial Attack on MNIST with VAE and Classifier

This project demonstrates an adversarial attack on the MNIST dataset using a Variational Autoencoder (VAE) and a classifier. The attack method used is the Fast Gradient Sign Method (FGSM).

## Introduction

In this project, the MNIST dataset is first transformed into latent codes using a Variational Autoencoder (VAE). The classifier is trained on these latent codes. The core idea is to compute the gradient of the classifier’s output with respect to the input image through the latent code. This is achieved by first calculating the gradient of the classifier’s output with respect to the latent code (∂y/∂z) and then calculating the gradient of the latent code with respect to the input image (∂z/∂x). By multiplying these two gradients (∂y/∂z * ∂z/∂x), we obtain the gradient of the classifier’s output with respect to the input image (∂y/∂x), which is then used for the FGSM attack.


