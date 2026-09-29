# 🌊 UAV Flood Segmentation using Deep Learning

A deep learning-based semantic segmentation project for detecting and segmenting flooded regions from **UAV (Unmanned Aerial Vehicle) aerial imagery**.

The project uses a **U-Net architecture with a pretrained ResNet34 encoder** to perform pixel-level binary segmentation of flood-affected regions. The complete workflow covers dataset preparation, image-mask validation, preprocessing, augmentation, model training, evaluation, visualization, and flood-area estimation.

---

## 📌 Project Overview

Floods can cause severe damage to infrastructure, agriculture, transportation, and communities. UAV imagery provides high-resolution aerial information that can be used to analyze flood-affected regions.

This project aims to automate the identification of flooded areas from UAV images using deep learning-based semantic segmentation.

### 🔄 Overall Pipeline

<img width="1774" height="887" alt="ChatGPT Image Sep 28, 2026, 12_00_30 AM" src="https://github.com/user-attachments/assets/05504d3c-dc64-480e-bc0f-e87f73676605" />





## 🖼️ Project Architecture

<img width="1774" height="887" alt="ChatGPT Image Sep 27, 2026, 11_33_08 PM" src="https://github.com/user-attachments/assets/f3d8e550-2453-4130-ae25-920a840602d6" />

---

## 🎯 Objectives

* Detect flooded regions from UAV aerial imagery.
* Perform pixel-level semantic segmentation of flood regions.
* Develop a binary flood segmentation model using U-Net.
* Utilize transfer learning through a pretrained ResNet34 encoder.
* Apply image augmentation to improve model generalization.
* Evaluate segmentation performance using IoU and Dice metrics.
* Visualize predicted segmentation masks against ground-truth masks.
* Estimate the percentage of an image covered by predicted flood regions.
* Categorize flood coverage based on the predicted flooded area.

---

# 🧠 Model Architecture

The project uses **U-Net** for semantic segmentation with **ResNet34** as the encoder.

### Model Configuration

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

The implementation uses the `segmentation-models-pytorch` framework.

---

## 🏗️ U-Net Architecture

U-Net follows an encoder-decoder architecture in which the encoder extracts hierarchical visual features while the decoder progressively reconstructs the spatial segmentation map.

The ResNet34 encoder is initialized with ImageNet pretrained weights to leverage learned visual representations.

```text
                    INPUT IMAGE
                         │
                         ▼
                ┌─────────────────┐
                │    ResNet34     │
                │     Encoder     │
                └────────┬────────┘
                         │
                  Feature Extraction
                         │
                         ▼
                ┌─────────────────┐
                │     Decoder     │
                │                 │
                │ Skip Connections│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Output Layer   │
                │  Binary Mask    │
                └────────┬────────┘
                         │
                         ▼
                  FLOOD / NON-FLOOD
```

---

## 📊 Dataset

The dataset used for this project was obtained from Kaggle.

🔗 **Dataset:** [Flood Area Segmentation – Kaggle](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)
The project works with UAV imagery paired with corresponding segmentation masks.

The dataset contains:

* UAV / aerial images
* Ground-truth segmentation masks
* Image-mask correspondence information through metadata

### Dataset Structure

```text
flood_dataset/
│
├── Image/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── Mask/
│   ├── mask_001.png
│   ├── mask_002.png
│   └── ...
│
└── metadata.csv
```

---

## 🔍 Dataset Validation

Before training, image-mask pairs are validated to ensure that:

* Images can be successfully loaded.
* Corresponding masks can be loaded.
* Image and mask dimensions are compatible.
* Invalid or unusable samples are excluded.

This helps prevent corrupted or incompatible samples from affecting model training.

---

## 📂 Dataset Split

The cleaned dataset is divided into:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        80% |
| Validation |        10% |
| Testing    |        10% |

A fixed random state is used to make the dataset split reproducible.

---

# 🔄 Data Preprocessing & Augmentation

The training pipeline applies image transformations to improve model robustness and generalization.

### Preprocessing

* Image resizing
* Mask resizing
* Normalization
* Tensor conversion

### Training Augmentation

The training pipeline includes transformations such as:

* Horizontal flipping
* Vertical flipping
* Random 90° rotation
* Brightness / contrast adjustment
* Shift
* Scaling
* Rotation
* Image normalization

Validation and testing data are processed without the training-specific augmentation operations.

---

## 🖼️ Data Augmentation Examples

<img width="1536" height="1024" alt="ChatGPT Image Sep 27, 2026, 11_52_32 PM" src="https://github.com/user-attachments/assets/6ed6310c-455a-4660-ab91-27f3957eef01" />


---
# ⚙️ Training

The model is trained using:

* U-Net architecture
* ResNet34 pretrained encoder
* Adam optimizer
* Combined BCE + Dice loss
* GPU acceleration when available

### Training Configuration

```text
Input Size       : 256 × 256
Batch Size       : 4
Epochs           : 25
Learning Rate    : 0.0001
Optimizer        : Adam
Architecture     : U-Net
Encoder          : ResNet34
````

The model tracks training and validation performance throughout the training process.

---

## 📉 Loss Function

The training objective combines **Binary Cross Entropy (BCE)** and **Dice Loss**.

```text
Total Loss = BCE Loss + Dice Loss
```

### Binary Cross Entropy

BCE evaluates pixel-level classification between the flood and non-flood classes.

### Dice Loss

Dice Loss focuses on the overlap between predicted and ground-truth segmentation regions.

The combination allows the model to optimize both pixel-level classification and segmentation overlap.

---

# 📈 Training Curves

The notebook generates training curves to monitor model learning, convergence, and validation performance.

The following figure combines the **training and validation loss, validation IoU, and validation Dice score** across the training epochs.

<img width="1189" height="390" alt="Training and Validation Curves" src="https://github.com/user-attachments/assets/ed6b12ef-a4a8-47b5-8283-48259eb62d1c" />

### 📊 Performance Trends

* **Loss:** Training and validation loss are monitored throughout training.
* **Validation IoU:** The best validation IoU reaches **0.7781**.
* **Validation Dice:** The best validation Dice score reaches **0.8633**.
* The curves provide an overview of the model's learning behavior and segmentation performance during training.

---

# 📊 Evaluation Metrics

The project evaluates segmentation performance using **Intersection over Union (IoU)** and **Dice Score**.

## Intersection over Union (IoU)

IoU measures the overlap between the predicted segmentation and the ground-truth mask.

```text
IoU = Intersection / Union
```

A higher IoU indicates greater overlap between the predicted and ground-truth regions.

---

## Dice Score

Dice measures the similarity between the predicted mask and ground-truth mask.

```text
Dice = 2 × Intersection / (Prediction + Ground Truth)
```

A higher Dice score indicates greater similarity between the predicted and ground-truth segmentation regions.

A prediction threshold of `0.5` is used for converting model probabilities into binary segmentation masks.

---

# 🏆 Model Performance

The final model is evaluated using **Intersection over Union (IoU)** and **Dice Score**.

### Final Evaluation

| Metric | Score |
|---|---:|
| Test IoU | **0.7941** |
| Test Dice Score | **0.8763** |
| Best Validation IoU | **0.7781** |
| Best Validation Dice | **0.8633** |

The best-performing model checkpoint is selected based on validation performance and subsequently evaluated on the held-out test dataset.

---

# 🖼️ Flood Segmentation Results

The trained model produces a binary segmentation mask identifying pixels classified as flooded.

The following example compares the original UAV imagery, ground-truth flood mask, and model prediction on an unseen test sample.

<!-- DRAG ONE GOOD NOTEBOOK PREDICTION OUTPUT HERE -->
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/b6260c53-495e-4839-9d50-a5162fc81829" />

<p align="center">
  <i>Comparison of original UAV imagery, ground-truth flood mask, and predicted flood segmentation.</i>
</p>

---

# 🌊 Flooded Area Estimation

The predicted segmentation mask is used to estimate the percentage of image pixels classified as flooded.

The calculation is:

```text
Flood Coverage (%) =
(Flood Pixels / Total Pixels) × 100
```

The predicted segmentation mask is thresholded at `0.5` before calculating the flooded coverage.

The resulting analysis contains:

```text
Image Name
Flood Coverage (%)
Risk Level
```

> Note: This represents pixel-based flood coverage within the image and is not a geographic area measurement such as hectares or square meters.

---

# ⚠️ Flood Coverage Categorization

The predicted flood coverage is categorized into three project-level levels:

| Flood Coverage | Risk Level |
|---|---|
| < 20% | Low |
| 20% – < 50% | Medium |
| ≥ 50% | High |

> These categories are project-level classifications based on predicted pixel coverage and should not be interpreted as official disaster-management standards.

---

## 📊 Flood Coverage Analysis

The segmentation predictions are further analyzed to estimate the percentage of each test image classified as flooded and assign a project-level risk category.

<!-- DRAG THE FLOOD COVERAGE DATAFRAME OUTPUT FROM THE NOTEBOOK HERE -->
<img width="456" height="237" alt="Screenshot 2026-09-28 000247" src="https://github.com/user-attachments/assets/05a43d7e-e7d5-4216-9d48-b75a5f7fdf95" />

<p align="center">
  <i>Flood coverage percentages and corresponding risk categories generated from test-set predictions.</i>
</p>

---

# 💾 Model Checkpoint

The best-performing model during training is saved as:

```text
best_model.pth
```

The checkpoint can be loaded using the same U-Net architecture and encoder configuration used during training.

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* U-Net
* ResNet34
* Transfer Learning
* Segmentation Models PyTorch

### Computer Vision

* Albumentations
* Pillow

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib

### Development Environment

* Google Colab
* Jupyter Notebook
* Visual Studio Code

---

# 📦 Installation

Install the main dependencies:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install timm
pip install albumentations==1.3.1
pip install numpy pandas matplotlib scikit-learn pillow tqdm
```

---

# 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vinayjoseph003/UAV-Flood-Segmentation.git
```

### 2. Navigate to the Project

```bash
cd UAV-Flood-Segmentation
```

### 3. Open the Notebook

Open:

```text
Flood_UAV.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* Visual Studio Code

### 4. Prepare the Dataset

Prepare the UAV flood dataset according to the dataset structure described above.

### 5. Update Dataset Paths

Update the dataset paths inside the notebook according to your execution environment.

### 6. Run the Notebook

Execute the notebook sequentially:

```text
Dataset Preparation
        ↓
Image / Mask Validation
        ↓
Dataset Split
        ↓
Data Augmentation
        ↓
Model Creation
        ↓
Model Training
        ↓
Validation
        ↓
Model Selection
        ↓
Test Evaluation
        ↓
Prediction Visualization
        ↓
Flood Area Estimation
```

---

# 🌍 Potential Applications

The developed segmentation pipeline can potentially be adapted for:

* UAV-based flood monitoring
* Flooded-area mapping
* Post-disaster assessment
* Emergency response support
* Agricultural flood monitoring
* Infrastructure assessment
* Remote sensing research
* Geospatial analysis
* Disaster management research
* Automated aerial image analysis

---

# 🔮 Future Improvements

Potential extensions of this project include:

* Experimenting with DeepLabV3+
* Experimenting with FPN and other segmentation architectures
* Comparing different pretrained encoders
* Hyperparameter optimization
* Cross-validation across different geographical regions
* Training on larger and more diverse UAV datasets
* Improved handling of class imbalance
* Advanced segmentation post-processing
* Geo-referenced flood-area estimation
* GIS integration
* Real-time UAV inference
* Edge-device deployment
* Web-based flood analysis dashboard
* Temporal flood monitoring using repeated UAV surveys

---

# 📌 Project Status

**Status: Research / Experimental Project**

The current repository contains the notebook-based implementation for UAV flood-water semantic segmentation, model training, evaluation, visualization, and flood-area estimation.

---

# 👨‍💻 Authors

## 1. Vinay Joseph Jonnakuti
## 2. Rani Chinthabathini
- GitHub : [ranichinthabathini-byte](https://github.com/ranichinthabathini-byte)

---

# 📜 License

This project is distributed under the license included in this repository.

See [`LICENSE`](LICENSE) for details.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and exploring the implementation in `Flood_UAV.ipynb`.

---
