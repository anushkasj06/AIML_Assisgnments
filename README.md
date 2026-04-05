# AIML_Assisgnments

# Deep Learning, Neural Networks & NLP Projects

This repository contains four comprehensive AI/ML projects implemented in Python using Jupyter Notebooks. These projects demonstrate the application of Fundamental Neural Networks, Advanced Transfer Learning, Object Detection, and Natural Language Processing techniques to solve real-world problems.

## 📂 Repository Contents

| File | Project Title | Description |
| :--- | :--- | :--- |
| `1_NeuralNetworks.ipynb` | **AQI Classification** | A Neural Network implementation to classify Air Quality Index (AQI) as 'Good' or 'Bad' based on environmental features. |
| `2_Transfer_Learning.ipynb` | **Brain Tumor Classification** | Leveraging pre-trained models (Transfer Learning) to detect and classify brain tumors from MRI scans. |
| `YOLOv8_Object_Detection_Lab_Completed.ipynb` | **Object Detection & Multi-Object Classification** | A comprehensive lab implementing YOLOv8 for object detection, instance segmentation, and image classification tasks. |
| `4_NLP_Text_Classification.ipynb` | **NLP Preprocessing & Text Classification** | Implements NLP preprocessing techniques and machine learning models for text classification tasks. |

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
This project implements **Transfer Learning** to enhance the classification of brain tumors from MRI images.

### Methodology
1. **Dataset Preparation:**
    * Utilizes the **TCGA-LGG dataset**
    * Image preprocessing and augmentation
2. **Model Architecture:**
    * Fine-tuning **VGG16, ResNet50, DenseNet121**
3. **Evaluation:**
    * Model comparison
    * EDA on patient demographics

### Libraries Used
* `tensorflow`
* `opencv`
* `sklearn`
* `PIL`
* `matplotlib`
* `seaborn`

---

## 🎯 Project 3: Object Detection & Multi-Object Classification with YOLOv8

### Overview
This project implements YOLOv8 for object detection, segmentation, and classification tasks.

### Key Features
* Object Detection (COCO128)
* Instance Segmentation
* Image Classification
* Benchmarking YOLOv8 variants
* Model export (ONNX, TorchScript)

### Libraries Used
* `ultralytics`
* `roboflow`
* `supervision`
* `opencv-python`
* `pandas`
* `numpy`

---

## 📝 Project 4: NLP Preprocessing & Text Classification

### Overview
This project focuses on building a complete **NLP pipeline** to preprocess textual data and classify it using machine learning algorithms.

### Key Features
* **Text Preprocessing**
  * Tokenization  
  * Stopword Removal  
  * Stemming & Lemmatization  

* **Vectorization Techniques**
  * CountVectorizer  
  * TF-IDF  

* **Machine Learning Models**
  * Naive Bayes  
  * Logistic Regression  
  * Support Vector Machine (SVM)  

* **Evaluation Metrics**
  * Accuracy  
  * Precision, Recall, F1-score  
  * Confusion Matrix  

* **Analysis**
  * Model comparison and performance insights  

### Libraries Used
* `nltk`
* `sklearn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

---

## 👥 Contributors

* **Anushka Sunil Jadhav**
* **Kiran Nandi**
* **Uday Sapate**
* **Om Panchal**

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/anushkasj06/AIML_Assisgnments.git
