# Car damge detection with segmentation
![example.png](https://github.com/user-attachments/assets/236c3d67-0f6e-400e-8bb8-0cfda1b7ae2c)

## Project Overview
This project was conducted as the **Final Project for the Sungkunkwan university 2024 Fall Semester DNN course**. Our team's topic was **Detect Car Damage - Segmentation**, and our goal is to design and improve a model to accurately segment damaged areas on vehicles.

You can see results and our work [here](https://round-vibraphone-01c.notion.site/Segmentation-156ccd25bf2280d68b32fe8de68bfbab?pvs=4)

## Dataset
- **AI Hub - Vehicle Damage Image Dataset** was sampled and used.
- Data split ratio: **Train:Test:Validation = 7:1.5:1.5**
- Image preprocessing: All images were resized to **512x512**.
- [차량 파손 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=581)
- [our sampled data with mask](https://drive.google.com/file/d/172xelPpQzVMdIT_SqK-MO9G1LOrFf38h/view?usp=drive_link)

## Baseline Models
We experimented with and compared various segmentation models. The following models were used:

1. **Mask2Former(b3)**
2. **UperNet for Semantic Segmentation (Swin-Tiny)**
3. **Segformer**
4. **DPT**
5. **MobileViTv2**
6. **DeepLabV3-MobileViT-Small**


## Model Improvement Methods
To enhance performance and efficiency, we implemented the following strategies:

1. **Ensemble Techniques**: Combining outputs from multiple models to achieve more accurate results.
2. **Loss Function Modification**: Customizing the loss function to optimize performance during training.
3. **Distillation**: Applied distillation with SegFomer. Teacher model : SegFomer(b3), Student model:SegFomer(B0)


## Results
Through this project, we successfully implemented models to effectively segment vehicle damage areas. By applying various improvement techniques, we enhanced both the performance and efficiency of the models. Especially, ensemble method worked well.

---

This project provided a valuable opportunity to explore practical applications of DNN techniques, while also discussing future directions for additional research and improvements.

