# SmartCrop: Intelligent Sports Video Reframing

**A Deep Learning solution for automating 16:9 to 9:16 video conversion by tracking the center of action.**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)
![Lightning](https://img.shields.io/badge/Lightning-2.6.0-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
In the era of vertical video platforms (TikTok, Instagram Reels, YouTube Shorts), converting standard broadcast footage to mobile formats is a manual bottleneck. **SmartCrop** automates this process by intelligently identifying and tracking the region of interest (ROI) in sports footage.

Unlike static center-cropping, this model uses a lightweight Convolutional Neural Network (CNN) to predict the optimal crop center in real-time, preserving the key action while discarding empty space.

### Key Features
* **Lightweight Architecture:** MobileNetV3 backbone optimized for edge deployment.
* **Spatial Softmax Head:** A differentiable spatial-to-numerical transform for precise coordinate regression, replacing standard Global Average Pooling.
* **Temporal Stability:** Implements an Exponential Moving Average (EMA) smoother with scene-cut detection to prevent camera jitter.
* **Uncertainty Modeling:** Outputs spatial probability maps (heatmaps) alongside coordinates, allowing for confidence-based filtering.

## Performance
The model achieves a **Mean IoU of ~0.68**, significantly outperforming standard heuristics.

| Method | Mean IoU | Improvement |
| :--- | :---: | :---: |
| Static Center Crop (Baseline) | 0.58 | - |
| **SmartCrop (Ours)** | **0.68** | **+17.25%** |

## Methodology

### 1. Model Architecture
Instead of treating the problem as a simple regression task (predicting a single float), I approached it as a density estimation problem to preserve spatial awareness.

* **Backbone:** `MobileNetV4` (Pretrained on ImageNet) extracts rich feature maps.
* **Neck:** 1x1 Convolutions reduce channel dimensionality while preserving the 7x7 spatial grid.
* **Head:** A fully convolutional upsampling decoder (7x7 $\to$ 28x28) with `LeakyReLU` activations and `BatchNorm`.
* **Output:** A **Spatial Softmax** layer converts the 2D heatmap into $(x, y)$ coordinates. This preserves spatial gradients better than fully connected layers and allows the network to handle multimodal distributions (e.g., two players on opposite sides).

### 2. Advanced Training Techniques
* **KL Divergence Loss:** Treated the target as a Gaussian distribution rather than a point. This allows the model to learn "areas of interest" rather than just single pixels.
* **Sigma Annealing:** Linearly decayed the target Gaussian sigma from $3.0$ (broad) to $1.5$ (sharp) during training. This helped the optimizer find the general valley early on while refining precision in later epochs.
* **Differential Learning Rates:** Fine-tuned the backbone at `1e-5` to preserve feature extraction capabilities, while training the regression head at `1e-4` for rapid adaptation.
