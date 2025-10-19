# 📘 Title: “Deep Neural Networks for Brain Tumor Detection”

---

## 🎯 Objective

This thesis will develop a system based on **Convolutional Neural Networks (CNNs)** for:

- **Classification** of MRI images into four categories:  
  *glioma, meningioma, pituitary, no tumor*  
- **Segmentation** of tumor regions in MRI images for visual representation and analysis.

---

## 🧠 Dataset: BRISC 2025

The **BRISC 2025** dataset is used, which includes:

- 6,000 **T1-weighted MRI slices**  
- Four **tumor categories**  
- **Pixel-wise segmentation masks**, verified by radiologists  
- Three **anatomical planes**: *Axial, Coronal, Sagittal*  
- Clean **train/test split** (5,000 / 1,000)  
- **Unique identifiers** for each slice, avoiding data leakage from the same patient  

---

## ⚙️ Methodology (Proposed)

1. **Preprocessing** of MRI slices (resize, normalization, augmentation)  
2. **Training CNN** for classification (e.g., *ResNet, EfficientNet*)  
3. **Training U-Net (or variant)** for segmentation  
4. *(Optional)* **Multi-task CNN** combining classification + segmentation  
5. **Evaluation** using metrics such as *accuracy, F1-score, IoU* for segmentation  
6. **Visualization** of results using *Grad-CAM* or heatmaps for explainability  

---

## 🏥 Clinical Value

- Supports radiologists in **early and accurate diagnosis**  
- Facilitates **tumor type classification** and **tumor region identification**  
- Reduces the likelihood of error and **saves time in clinical practice**

---

## 🌟 Significance

- Utilizes a **high-quality, expert-annotated dataset** that surpasses older datasets (*Figshare, SARTAJ*)  
- Enables **advanced research** in **multi-task deep learning**

---
