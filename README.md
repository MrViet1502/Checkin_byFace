# 👤 Face Recognition Check-in System

A real-time face recognition-based check-in system developed using **Python**, **Flask**, **OpenCV**, and **KNN algorithm**. The system allows users to register, verify their identity via webcam, and automatically logs check-in data through a web interface.

## 🔧 Features

- Register users with personal info and facial images
- Train a KNN model for face recognition using `scikit-learn` and `face_recognition`
- Real-time face detection and identity verification via webcam
- Automatic check-in logging into CSV files
- Web interface for registration and viewing check-in history

## 📌 Technologies

- Python
- Flask
- OpenCV
- scikit-learn
- face_recognition
- HTML/CSS/JS (basic)

## 📁 Project Structure

Checkin_byFace/ ├── knn_folder/ │ ├── train/ # Registered face images by user │ └── trained_knn_model.clf ├── templates/ │ └── index.html # Web UI for registration and check-in list ├── static/ # Frontend styling and JS ├── app.py # Main app logic ├── user_info.csv # User registration info └── checkin_log.csv # Check-in history

## 🚀 How It Works

1. **User Registration:**  
   - Users submit name, date of birth, phone number, and upload a face image.  
   - Image and info are stored locally and used to train/update the KNN model.

2. **Face Recognition (Check-in):**  
   - Webcam captures real-time frames and compares detected face encodings with the trained model.  
   - If matched, info is logged to `checkin_log.csv` and displayed on the screen.

3. **View Check-ins:**  
   - A web interface shows the list of users who have checked in, with timestamps.

## 📷 Demo

> Demo available in report file

## 🛠 Installation

```bash
  pip install -r requirements.txt
  python app.py
```
Make sure to install dependencies: Flask, OpenCV, face_recognition, scikit-learn, etc.
