# 🌊 UAV Flood Segmentation using Deep Learning

A deep learning-based semantic segmentation project for detecting and segmenting flooded regions from **UAV (Unmanned Aerial Vehicle) aerial imagery**.

The project uses a **U-Net architecture with a pretrained ResNet34 encoder** to perform pixel-level binary segmentation of flood-affected regions. The complete workflow covers dataset preparation, image-mask validation, preprocessing, augmentation, model training, evaluation, visualization, and flood-area estimation.

---

## 📌 Project Overview

Floods can cause severe damage to infrastructure, agriculture, transportation, and communities. UAV imagery provides high-resolution aerial information that can be used to analyze flood-affected regions.

This project aims to automate the identification of flooded areas from UAV images using deep learning-based semantic segmentation.

### 🔄 Overall Pipeline


## 🖼️ Project Architecture
<!-- ADD GENERATED ARCHITECTURE IMAGE HERE --> <!-- Suggested filename: assets/architecture.png --> <p align="center"> <img src="assets/architecture.png" alt="UAV Flood Segmentation Architecture" width="900"> </p> <p align="center"> <i>End-to-end architecture of the UAV flood segmentation system.</i> </p>

## 🎯 Objectives
Detect flooded regions from UAV aerial imagery.
Perform pixel-level semantic segmentation of flood regions.
Develop a binary flood segmentation model using U-Net.
Utilize transfer learning through a pretrained ResNet34 encoder.
Apply image augmentation to improve model generalization.
Evaluate segmentation performance using IoU and Dice metrics.
Visualize predicted segmentation masks against ground-truth masks.
Estimate the percentage of an image covered by predicted flood regions.
Categorize flood coverage based on the predicted flooded area.

## 🧠 Model Architecture

The project uses U-Net for semantic segmentation with ResNet34 as the encoder.

| Component        | Configuration                    |
| ---------------- | -------------------------------- |
| Architecture     | U-Net                            |
| Encoder          | ResNet34                         |
| Encoder Weights  | ImageNet pretrained              |
| Input Channels   | 3                                |
| Output Classes   | 1                                |
| Task             | Binary Semantic Segmentation     |
| Input Resolution | 256 × 256                        |
| Optimizer        | Adam                             |
| Learning Rate    | 0.0001                           |
| Epochs           | 25                               |
| Batch Size       | 4                                |
| Device           | CUDA if available, otherwise CPU |


The implementation uses the segmentation-models-pytorch framework.

## 🏗️ U-Net Architecture

U-Net follows an encoder-decoder architecture:


## 📊 Dataset

The project works with UAV imagery paired with corresponding segmentation masks.

The dataset contains:

UAV/aerial images
Ground-truth segmentation masks
Image-mask correspondence information through metadata

## 🔍 Dataset Validation

Before training, image-mask pairs are validated to ensure that:

Images can be successfully loaded.
Corresponding masks can be loaded.
Image and mask dimensions are compatible.
Invalid or unusable samples are excluded.

This helps prevent corrupted samples from affecting model training.

## 📂 Dataset Split

The cleaned dataset is divided into:
| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        80% |
| Validation |        10% |
| Testing    |        10% |
A fixed random state is used to make the split reproducible.

## 🔄 Data Preprocessing & Augmentation

The training pipeline applies image transformations to improve model robustness.

Preprocessing
Image resizing
Mask resizing
Normalization
Tensor conversion
Training Augmentation

The training pipeline includes transformations such as:

Horizontal flipping
Vertical flipping
Random 90° rotation
Brightness/contrast adjustment
Shift
Scaling
Rotation
Image normalization

Validation and testing data are processed without the training-specific augmentations.

## 🖼️ Data Augmentation Examples


## ⚙️ Training

The model is trained using:

U-Net
ResNet34 pretrained encoder
Adam optimizer
Combined BCE + Dice loss
GPU acceleration when available
Training Configuration:
Input Size       : 256 × 256
Batch Size       : 4
Epochs           : 25
Learning Rate    : 0.0001
Optimizer        : Adam
Architecture     : U-Net
Encoder          : ResNet34
The model tracks training and validation performance throughout the training process.

## 📈 Training Curves

The notebook generates training curves to monitor model learning and convergence.


## 📊 Evaluation Metrics

The project evaluates segmentation performance using:

Intersection over Union (IoU)

IoU measures the overlap between the predicted segmentation and the ground-truth mask.

IoU = Intersection / Union
Dice Score

Dice measures the similarity between the predicted mask and ground-truth mask.

Dice = 2 × Intersection / (Prediction + Ground Truth)

A prediction threshold of 0.5 is used for converting model probabilities into binary segmentation masks.

## 🏆 Model Performance
Final Evaluation
| Metric          |      Score |
| --------------- | ---------: |
| Test IoU        | Add result |
| Test Dice Score | Add result |
| Validation IoU  | Add result |
| Validation Dice | Add result |

## 🛠️ Technologies Used
Programming
Python
Deep Learning
PyTorch
U-Net
ResNet34
Transfer Learning
Segmentation Models PyTorch
Computer Vision
Albumentations
Pillow
Data Processing
NumPy
Pandas
Scikit-learn
Visualization
Matplotlib
Development Environment
Google Colab
Jupyter Notebook
Visual Studio Code
