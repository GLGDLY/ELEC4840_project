# SwinUNERT Model

This is the README file for our approach 1 inspired by SwinUNETR

# Overview
This notebook implements our improved U-Net model architecture with attention gate and residual blocks for 3D medical spleen CT-images segmentation. This model combines Focal Tversky loss and Dice loss for training and evaluation.

# Features
- Enhanced U-Net Model Structure
    - Residual Blocks with SE Modules
    - Attention Gates between Encoder and Decoder Path
    - Deep supervision with Skip Connection
- Loss Function
    - Combined use of Focal Tversky Loss and Dice Loss

# Dataset
This model is trained on the provided Spleen Dataset:
- 25 Training Images + Masks
- 8 Validation Images + Masks
- 3 Test Images + Masks

# Performance
The best validation Dice Score achieved was at epoch 68 The result on the test set:
- Dice Coefficient: 0.76
- Jaccard Coefficient: 0.64

# Requirement
- Python 3.8
- PyTorch
- MONAI
- SimpleITK
- MedPy
- Nibabel
- TensorBoard