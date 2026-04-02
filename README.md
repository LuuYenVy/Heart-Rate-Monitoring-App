# ❤️ HeartGuard - Heart Rate Monitoring & Prediction App

## 📌 Overview

**HeartGuard** is a mobile application designed to monitor heart rate data and predict potential heart disease risks using machine learning models.

The app helps users track their health, detect abnormalities early, and manage medical appointments efficiently.

---
You can view the demo here:
---

## 🚀 Features

* 📊 Real-time heart rate tracking
* 🤖 Heart disease risk prediction using AI
* 📅 Appointment scheduling and management
* 🔔 Abnormal heart rate alerts
* 🔐 User authentication (Firebase Auth)
* ☁️ Cloud data storage (Firebase Realtime Database)

---

## 🛠 Tech Stack

### 📱 Mobile

* Android (Java)
* MVVM Architecture
* ViewModel + LiveData

### 🔥 Backend & Cloud

* Firebase Realtime Database
* Firebase Authentication

### 🤖 AI / Backend API

* Python (FastAPI)
* RESTful API
* Machine Learning model (for prediction)

### 🧰 Tools

* Android Studio
* Postman
* Git & GitHub

---

## 🏗 Architecture

The application is built using **MVVM (Model - View - ViewModel)** combined with the **Repository pattern**.

### 🔄 Data Flow

User Interaction (UI)
→ ViewModel (handles UI logic & state)
→ Repository (single source of truth)
→ Data Sources:

* Firebase Realtime Database (user data, appointments)
* FastAPI Backend (ML prediction API)

### 🧩 Components

* **View (Activity/Fragment)**
  Handles UI rendering and user interactions

* **ViewModel**
  Manages UI-related data using LiveData and survives configuration changes

* **Repository**
  Abstracts data sources and provides clean API for ViewModel

* **Data Sources**

  * Firebase: stores user data in real-time
  * Backend API: processes heart rate data and returns prediction results

### 📌 Key Benefits

* Separation of concerns
* Easier testing and maintenance
* Scalable architecture for future features

---

## 📷 Screenshots

> *(Add your app screenshots here)*

* Login Screen
* Dashboard
* Heart Rate Chart
* Prediction Result

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/heartguard.git
```

### 2. Open project

* Open with **Android Studio**

### 3. Firebase setup

* Add your `google-services.json` file to:

```
app/
```

### 4. Run backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

---

## 🔌 API Documentation

### POST `/predict`

#### Request

```json
{
  "heartRate": [72, 75, 80, 78],
  "age": 25
}
```

#### Response

```json
{
  "risk": "low"
}
```

---


## 🧪 Testing

* Manual testing on Android device/emulator
* API testing using Postman

---

## 📌 Future Improvements

* ⏳ Integrate wearable devices (smartwatch, IoT)
* 📈 Improve ML model accuracy
* ☁️ Deploy backend to cloud (AWS / Render / Railway)
* 📊 Advanced data visualization dashboard

---

## 👤 Author

* GitHub: https://github.com/your-username
* TikTok: https://www.tiktok.com/@umaru9195

---

## ⭐ Acknowledgements

This project is developed for learning purposes in:

* Mobile Development
* Data Analytics
* Machine Learning Integration

---

## 💡 Notes

This project demonstrates a combination of:

* Mobile App Development (Android)
* Backend API Development
* Data-driven health prediction

---

⭐ If you find this project useful, feel free to give it a star!
