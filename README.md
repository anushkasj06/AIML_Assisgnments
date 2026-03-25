# AIML_Assisgnments

# Deep Learning & Neural Networks Projects

This repository contains three comprehensive Deep Learning projects implemented in Python using Jupyter Notebooks. These projects demonstrate the application of Fundamental Neural Networks, Advanced Transfer Learning, and Object Detection techniques to solve real-world problems.

## 📂 Repository Contents

| File | Project Title | Description |
| :--- | :--- | :--- |
| `1_NeuralNetworks.ipynb` | **AQI Classification** | A Neural Network implementation to classify Air Quality Index (AQI) as 'Good' or 'Bad' based on environmental features. |
| `2_Transfer_Learning.ipynb` | **Brain Tumor Classification** | Leveraging pre-trained models (Transfer Learning) to detect and classify brain tumors from MRI scans. |
| `YOLOv8_Object_Detection_Lab_Completed.ipynb` | **Object Detection & Multi-Object Classification** | A comprehensive lab implementing YOLOv8 for object detection, instance segmentation, and image classification tasks. |

---

## 🧪 Project 1: Neural Networks Design and Deployment (AQI Classification)

### Overview
This project focuses on designing and deploying a Neural Network to predict Air Quality. It involves binary classification to determine if the AQI is "Good" or "Bad" based on input sensor data.

### Key Features
* **Data Processing:** Handling environmental datasets using **Pandas** and **NumPy**.
* **Model Architecture:** Implementation of a neural network for binary classification.
* **Evaluation:**
    * **Confusion Matrix:** Visualizing True Positives/Negatives vs. False Positives/Negatives.
    * **Confidence Distribution:** Analyzing the model's probability scores (Sigmoid output) to understand prediction certainty.
* **Visualization:** Using **Seaborn** and **Matplotlib** for heatmaps and distribution plots.

### Libraries Used
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `tensorflow` / `keras`

---

## 🧠 Project 2: Transfer Learning for Brain Tumor Classification

### Overview
This project implements **Transfer Learning** to enhance the classification of brain tumors from MRI images. It utilizes state-of-the-art pre-trained architectures to achieve high accuracy in distinguishing between tumor and non-tumor regions.

### Methodology
1.  **Dataset Preparation:**
    * Utilizes the **TCGA-LGG (Lower Grade Glioma)** dataset containing patient MRI scans and segmentation masks.
    * Preprocessing involves image resizing, mask handling, and data augmentation.
2.  **Model Architecture:**
    * **Transfer Learning:** Fine-tuning pre-trained models such as **VGG16**, **ResNet50**, and **DenseNet121**.
    * **Segmentation (Res-U-Net):** References to Res-U-Net for understanding tumor regions via image segmentation masks.
3.  **Evaluation:**
    * Comparative analysis of model performance.
    * Exploratory Data Analysis (EDA) on patient demographics (Age, Gender, Tumor Location).

### Libraries Used
* `tensorflow` / `keras`
* `cv2` (OpenCV)
* `sklearn` (scikit-learn)
* `PIL` (Pillow)
* `matplotlib` & `seaborn`

---

## 🎯 Project 3: Object Detection & Multi-Object Classification with YOLOv8

### Overview
This project implements YOLOv8 for object detection, instance segmentation, and image classification. It covers training, evaluation, benchmarking, and deployment of YOLOv8 models on various tasks.

### Key Features
* **Object Detection:** Training and evaluating YOLOv8 detection models on datasets like COCO128.
* **Instance Segmentation:** Using YOLOv8-seg for pixel-level object segmentation.
* **Image Classification:** Leveraging YOLOv8-cls for top-k classification.
* **Benchmarking:** Comparing different YOLOv8 variants (n, s, m) for speed and accuracy.
* **Fine-Tuning:** Hyperparameter optimization and data augmentation.
* **Export & Deployment:** Exporting models to ONNX and TorchScript for deployment.

### Libraries Used
* `ultralytics`
* `roboflow`
* `supervision`
* `matplotlib`
* `seaborn`
* `pandas`
* `numpy`
* `opencv-python`
* `pyyaml`
* `pillow`

---

## 👥 Contributors

* **Anushka Sunil Jadhav**
* **Kiran Nandi**
* **Uday Sapate**
* **Om Panchal**

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anushkasj06/AIML_Assisgnments.git](https://github.com/anushkasj06/AIML_Assisgnments.git)
    ```
2.  **Open in Jupyter or Google Colab:**
    * Upload the `.ipynb` files to [Google Colab](https://colab.research.google.com/).
    * Ensure the required datasets are mounted (for the Brain Tumor project, the code expects data in Google Drive).
3.  **Install Dependencies:**
    ```python
    !pip install tensorflow numpy pandas matplotlib seaborn opencv-python ultralytics roboflow supervision pyyaml pillow
    ```

## 📜 License
This project is for educational and academic purposes.
