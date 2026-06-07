# MNIST Neural Architecture Audits & Mechanism Analysis

This repository contains a structured series of deep learning diagnostic experiments on the MNIST dataset using PyTorch. Instead of just aiming for high accuracy, these modules evaluate the core mathematical mechanics of neural networks, including feature bottlenecks, regularizations, hyperparameter scheduling, and architectural evolution.

## Repository Pipeline

- **`mnist_linear_baseline.ipynb`**: Establishes the performance baseline using a single-layer linear model (7,850 parameters). Achieved 91.03% test accuracy with standard SGD.
- **`network_width_bottleneck_analysis.ipynb`**: Explores the *Information Bottleneck Effect* by restricting hidden layer widths ($k$) under controlled parameters ($P = 10^4, 5\times10^4, 10^5$). Evaluates convergence boundaries when information pathways approach zero.
- **`weight_decay_regularization.ipynb`**: Implements L2 regularization (Weight Decay = 1e-3) within flexible dense architectures to quantitatively audit overfitting suppression.
- **`optimizer_activation_grid_search.ipynb`**: Performs a comprehensive grid search across 8 hyperparameter combinations (SGD vs. Adam $\times$ Sigmoid, Tanh, ReLU, ELU), tracking loss landscapes and convergence speeds into structured Pandas DataFrames.
- **`dense_vs_cnn_architecture_search.ipynb`**: Investigates spatial feature extraction by transitioning from Multilayer Perceptrons (MLP) to Convolutional Neural Networks (CNN). Implements an automated constraints-driven search to discover the minimum parameter footprint that matches baseline dense capacities.

## Tech Stack & Dependencies
- **Core**: PyTorch, TorchVision
- **Analysis & Visualization**: NumPy, Pandas, Matplotlib, Seaborn
