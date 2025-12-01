# Pneumonia Detection

## Overview
This repository contains a complete **end-to-end deep learning
pipeline** for detecting **pneumonia from chest X-ray images** using
modern convolutional neural networks and GAN-based augmentation.

The project includes:

-   **Transfer learning models** (EfficientNetB0, DenseNet121, CheXNet,
    ResNet50)
-   **Two-phase training** (frozen base → fine-tuning)
-   **Dataset preprocessing & augmentation**
-   **Automatic class balancing** using classical augmentation
-   **Optional GAN-based synthetic image generation**
-   **Evaluation tools** (confusion matrix, ROC curve, PR curve,
    metrics, reports)
-   **Model checkpointing, history plots, and result saving**

This project is structured for reproducible medical-imaging experiments
and can be used both for research and for production-grade training
workflows.

------------------------------------------------------------------------

## 📦 Repository Structure

    ├── preprocessing.py                # Dataset analysis, visualization, augmentation
    ├── balancing.py                    # Classical augmentation-based balancing
    ├── transfer_learning_classifier.py # Transfer learning architectures and training logic
    ├── train_transfer_models.py        # Main script for training CNN models
    ├── gan_generator.py                # Script for generating synthetic X-ray images using GAN
    ├── models/                         # Saved models (.h5)
    ├── checkpoints/                    # Best-model checkpoints
    ├── results/                        # Confusion matrix, ROC, PR curves, metrics
    └── chest_xray/                     # Expected dataset structure (train/val/test)

------------------------------------------------------------------------

# ⚙️ Requirements

-   **Python 3.12**
-   TensorFlow / Keras
-   NumPy, Matplotlib, Seaborn
-   Scikit-learn
-   PIL / OpenCV
-   tqdm

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📁 Dataset Structure

The project expects the Chest X-Ray dataset in the following format:

    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

You may use the Kaggle Chest X-Ray Pneumonia dataset or any dataset
matching this structure.

------------------------------------------------------------------------

# 🚀 How to Run the Project

There are **two main runnable scripts**:

### 1. Train a Transfer Learning Model

**File:** `train_transfer_models.py`

This script:

-   Analyzes the dataset
-   Creates augmentation generators
-   Optionally balances the dataset
-   Builds and trains a selected model
-   Saves:
    -   best checkpoint
    -   final model
    -   training history plots
    -   confusion matrix / ROC / PR curves
    -   metrics JSON
    -   classification report

### **Usage**

``` bash
python3.12 train_transfer_models.py --model densenet121 --dataset chest_xray
```

### **Supported models**

-   `densenet121`
-   `efficientnetb0`
-   `resnet50`
-   `chexnet` (DenseNet121 with pretrained CheXNet weights)

------------------------------------------------------------------------

### 2. Generate Synthetic Images Using GAN

**File:** `gan_generator.py`

This script loads a trained GAN **generator** and produces synthetic
X-ray images (grayscale) for augmentation.

### **Usage**

``` bash
python3.12 gan_generator.py \
            --model path/to/generator.h5 \
            --num_images 2500 \
            --output_dir gan-generated \ 
            --upscale 224 \
            --prefix gan_ 
```

Generated images can then be copied to:

    chest_xray/val/NORMAL/

or any other class folder depending on your needs.

------------------------------------------------------------------------


