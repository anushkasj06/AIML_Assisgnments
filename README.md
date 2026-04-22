# AIML_Assisgnments

# Deep Learning, Neural Networks & NLP Projects

This repository contains five comprehensive AI/ML projects implemented in Python using Jupyter Notebooks and FastAPI. These projects demonstrate the application of Fundamental Neural Networks, Advanced Transfer Learning, Object Detection, Natural Language Processing techniques, and API deployment to solve real-world problems.

## 📂 Repository Contents

| File | Project Title | Description |
| :--- | :--- | :--- |
| `1_NeuralNetworks.ipynb` | **AQI Classification** | A Neural Network implementation to classify Air Quality Index (AQI) as 'Good' or 'Bad' based on environmental features. |
| `2_Transfer_Learning.ipynb` | **Brain Tumor Classification** | Leveraging pre-trained models (Transfer Learning) to detect and classify brain tumors from MRI scans. |
| `YOLOv8_Object_Detection_Lab_Completed.ipynb` | **Object Detection & Multi-Object Classification** | A comprehensive lab implementing YOLOv8 for object detection, instance segmentation, and image classification tasks. |
| `Assignment4_NLP_Text_Classification.ipynb` | **NLP Preprocessing & Text Classification** | Implements NLP preprocessing techniques and machine learning models for text classification tasks. |
| `TextPrediction/` | **Next Word Prediction API** | An LSTM-based text prediction model deployed as a FastAPI web service for predicting the next word in a sequence. |

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


## 🔄 Assignment 6: Encoder-Decoder with BiLSTM

### Overview
This assignment implements an Encoder-Decoder architecture using Bidirectional LSTM (BiLSTM) for sequence-to-sequence tasks, such as text translation or sequence prediction. The project demonstrates how encoder-decoder models can learn to map input sequences to output sequences, a foundational concept in modern NLP.

### Key Features
* **Encoder-Decoder Architecture:** Utilizes BiLSTM layers for both encoding and decoding sequences.
* **Sequence-to-Sequence Learning:** Handles variable-length input and output sequences.
* **Attention Mechanism (if implemented):** Optionally includes attention for improved context handling.
* **Evaluation:** Demonstrates model performance on relevant sequence tasks.

### Libraries Used
* `tensorflow` / `keras`
* `numpy`
* `pandas`
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

## � Project 5: Next Word Prediction API with FastAPI

### Overview
This project implements an LSTM-based neural network for next word prediction and deploys it as a RESTful API using FastAPI. The model is trained on text data from "Alice's Adventures in Wonderland" and can predict the most likely next word given an input sequence.

### Key Features
* **Model Training:** LSTM neural network trained for sequence prediction
* **API Deployment:** FastAPI web service with automatic documentation
* **Text Preprocessing:** Tokenization and sequence padding
* **Prediction Endpoint:** POST `/predict` for next word prediction
* **Interactive Documentation:** Built-in Swagger UI at `/docs`

### Architecture
* **Input Processing:** Tokenizes input text and pads sequences to fixed length
* **Model:** LSTM layers with embedding for word prediction
* **Output:** Returns predicted next word and complete text sequence

### Libraries Used
* `fastapi`
* `uvicorn`
* `tensorflow`
* `numpy`
* `pydantic`

### How to Run the API
1. Navigate to the `TextPrediction/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn main:app --reload`
4. Access the API at `http://127.0.0.1:8000`
5. View documentation at `http://127.0.0.1:8000/docs`

---

## �👥 Contributors

* **Anushka Sunil Jadhav**
* **Kiran Nandi**
* **Uday Sapate**
* **Om Panchal**

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/anushkasj06/AIML_Assisgnments.git
