# PGN_PyTorch
This repository provides a PyTorch implementation related to human parsing, inspired by the Part Grouping Network (PGN) approach. While not a direct reimplementation of the full PGN architecture from the paper "[Instance-level Human Parsing: Beyond Part Pooling](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_Gong_Instance-level_Human_Parsing_ECCV_2018_paper.pdf)" (ECCV 2018), it explores key concepts and techniques relevant to human part segmentation using U-Net. The original PGN was implemented in TensorFlow; this project aims to provide a PyTorch-based exploration of related segmentation techniques.

**Project Overview**

This project focuses on building and training a U-Net model for human part segmentation. It includes explorations of fundamental concepts like transposed convolutions and building a U-Net architecture from scratch. The primary dataset used is CIHP-CelebMaskHQ-6class-256 (available on Kaggle: [https://www.kaggle.com/datasets/remainaplomb/cihp-celebmaskhq-6class-256](https://www.kaggle.com/datasets/remainaplomb/cihp-celebmaskhq-6class-256)).

**Getting Started**

*   **Prerequisites:**

    *   Python 3.7+
    *   PyTorch (tested with 1.12.1 or later)
    *   torchvision
    *   tqdm
    *   PIL (Pillow)
    *   matplotlib
    *   NumPy
    *   OpenCV (cv2)
    *   Kaggle API (for dataset download if not manually downloaded)
    *   TensorBoard (optional, for visualization)
   
 **Notebooks**

This repository contains the following Jupyter Notebooks:

1.  **Transposed Convolution In Pytorch.ipynb:** This notebook demonstrates the concept of transposed convolution (deconvolution) in PyTorch. It includes both a manual implementation for educational purposes and the use of PyTorch's built-in `nn.ConvTranspose2d` module for practical application.

2.  **U_Net_from_Scratch.ipynb:** This notebook implements the U-Net architecture from scratch using PyTorch. It defines the building blocks of the U-Net, including double convolutions, downsampling (max pooling), and upsampling (transposed convolution with skip connections). It also includes a basic test with a dummy input tensor.

3.  **U_Net_CIHP_Segmentation.ipynb:** This is the core notebook for human part segmentation. It uses the U-Net architecture implemented in the previous notebook and applies it to the CIHP-CelebMaskHQ-6class-256 dataset. It covers:

    *   Data loading and preprocessing using a custom `Dataset` class.
    *   Model training with Cross-Entropy Loss, Dice Loss, and IoU Loss.
    *   Model evaluation on a test set.
    *   Visualization of results and TensorBoard integration for monitoring training progress.

**Data**

The CIHP-CelebMaskHQ-6class-256 dataset is used for training and testing. Due to its size, it is *not* included in the repository. You can download it from Kaggle: [https://www.kaggle.com/datasets/remainaplomb/cihp-celebmaskhq-6class-256](https://www.kaggle.com/datasets/remainaplomb/cihp-celebmaskhq-6class-256).
![image](https://github.com/user-attachments/assets/c692b705-b97f-48fb-8557-6b207e6e9cef)
![image](https://github.com/user-attachments/assets/4052ab6c-9e4b-4ddc-bfc8-2d840ca0dede)


**Further Development**

This project can be extended in several ways:

*   Implement the full PGN architecture as described in the original paper.
*   Experiment with different loss functions, optimizers, and hyperparameters.
*   Evaluate the model on other human parsing datasets.
*   Implement data augmentation techniques to improve model robustness.
