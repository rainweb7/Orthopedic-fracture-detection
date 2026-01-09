# 🦴 AI-Based Orthopedic Fracture Detection System

## 📌 Overview
This project is an **AI-assisted non-invasive orthopedic fracture detection system** that analyzes X-ray images using deep learning techniques.  
The system acts as a **decision-support software tool** to assist medical professionals by predicting the presence of bone fractures from uploaded images.

> ⚠️ Note: This system is designed for educational and research purposes only and does not replace professional medical diagnosis.

---

## 🎯 Problem Statement
Manual analysis of orthopedic X-ray images is time-consuming and prone to human error.  
There is a need for a **software-based intelligent system** that can assist in detecting fractures quickly and consistently using machine learning.

---

## 💡 Proposed Solution
The proposed system uses a **Convolutional Neural Network (CNN)** with **transfer learning (MobileNetV2)** to analyze X-ray images and classify them as:
- **Fracture Detected**
- **No Fracture**

The trained model is integrated into a **Flask-based web application**, allowing users to upload images and receive predictions through a simple interface.

---

## 🧠 System Architecture
1. User uploads X-ray image via web interface  
2. Image preprocessing using OpenCV  
3. Trained CNN model performs prediction  
4. Result and confidence score are displayed  

---

## 🛠 Technology Stack

### Programming & Frameworks
- Python 3.9+
- TensorFlow (Keras API)
- Flask

### Libraries
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### Frontend
- HTML
- CSS (basic)

### Tools
- VS Code
- Git & GitHub
- Docker (optional)

---

## 📂 Project Structure

