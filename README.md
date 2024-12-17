# DNN_project
Detect Car damge - segmentation

## Project Overview
This project was conducted as the **Final Project for the 2024 Fall Semester DNN course**. Our team's topic was **Detect Car Damage - Segmentation**, with the goal of designing and improving a model to accurately segment damaged areas on vehicles.

## Dataset
- **AI Hub Vehicle Damage Image Dataset** was sampled and used.
- Data split ratio: **Train:Test:Validation = 7:1.5:1.5**
- Image preprocessing: All images were resized to **512x512**.

## Baseline Models
We experimented with and compared various segmentation models. The following models were used:

1. **Mask2Former**
2. **UperNet for Semantic Segmentation (Swin-Tiny)**
3. **Segformer**
4. **DPT**
5. **MobileViTv2**
6. **DeepLabV3-MobileViT-Small**

## Model Improvement Methods
To enhance performance and efficiency, we implemented the following strategies:

1. **Ensemble Techniques**: Combining outputs from multiple models to achieve more accurate results.
2. **Loss Function Modification**: Customizing the loss function to optimize performance during training.
3. **Model Lightweighting**: Leveraging lightweight models such as MobileViT and Swin-Tiny, with additional optimizations for reduced model size.

## Results
Through this project, we successfully implemented models to effectively segment vehicle damage areas. By applying various improvement techniques, we enhanced both the performance and efficiency of the models.

---

This project provided a valuable opportunity to explore practical applications of DNN techniques, while also discussing future directions for additional research and improvements.

