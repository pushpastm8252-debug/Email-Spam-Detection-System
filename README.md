# 📧 Email Spam Detection System

An ML-powered Spam Detection System built using Python, Streamlit, Scikit-learn, and SQLite. The application predicts whether a message is **Spam** or **Not Spam** using a trained Machine Learning model.

> **Note:** The current model is trained on the UCI SMS Spam Collection dataset. It currently demonstrates SMS spam classification, not validated email-specific classification.

## 🚀 Live Demo

**Live Application:** https://email-spam-detection-system-9135.streamlit.app/

**GitHub Repository:** https://github.com/pushpastm8252-debug/Email-Spam-Detection-System

## 📌 Project Overview

Spam messages are unwanted or misleading messages that can contain advertisements, fraudulent offers, or suspicious links.

This project uses Machine Learning algorithms to classify messages into two categories:

- Spam
- Not Spam (Ham)

The system provides an interactive Streamlit interface where users can test messages and explore model performance.

## ✨ Features

- Spam and Not Spam message classification
- Machine Learning-based prediction
- Multiple ML algorithm comparison
- Model performance analytics
- Confusion matrix visualization
- Prediction history using SQLite
- Interactive Streamlit dashboard
- Developer profiles section

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Streamlit | Web application |
| SQLite | Prediction history database |
| Joblib | Model serialization |
| Git & GitHub | Version control |
| Streamlit Cloud | Deployment |

## 🤖 Machine Learning Models

Three Machine Learning algorithms were trained and evaluated:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Naive Bayes | 96.32% | 82.57% |
| Logistic Regression | 96.32% | 82.73% |
| SVM | 98.55% | 93.98% |

**Best Performing Model: Support Vector Machine (SVM)**

The SVM model achieved the highest accuracy and F1 score among the three evaluated models.

## 📊 Model Evaluation

The project evaluates model performance using:

- Accuracy
- F1 Score
- Confusion Matrix

These metrics help assess classification performance, including how well spam and legitimate messages are identified.

## 📂 Project Structure

```text
Email-Spam-Detection-System/
│
├── assets/
│   └── developers/
│       ├── mohit_kumar.jpg
│       └── pushpa_kumari.jpg
│
├── database/
│
├── dataset/
│   ├── SMSSpamCollection
│   ├── raw_spam_data.csv
│   └── cleaned_spam_data.csv
│
├── models/
│   ├── best_spam_model.joblib
│   ├── model_metrics.json
│   └── confusion_matrix.png
│
├── notebooks/
│
├── app.py
├── developers_section.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pushpastm8252-debug/Email-Spam-Detection-System.git
```

### 2. Navigate to Project Folder

```bash
cd Email-Spam-Detection-System
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🧪 Testing Examples

**Example 1: Spam Message**

```text
Congratulations! You have won a free prize. Click now!
```

Expected category: Spam

**Example 2: Normal Message**

```text
Hey, are we meeting tomorrow at college?
```

Expected category: Not Spam

*Predictions depend on the trained model.*

## 📈 Future Improvements

- Train and validate the model using a dedicated email spam dataset.
- Add email file upload and email content analysis.
- Improve text preprocessing and feature extraction.
- Add more Machine Learning and Deep Learning models.
- Add explainable AI for prediction insights.
- Improve dashboard analytics and user experience.
- Add secure user authentication.

## 👩‍💻 Development Team

### Pushpa Kumari
**Role:** AI/ML & Python Developer

- Python and Machine Learning
- Generative AI and RAG
- Streamlit Application Development
- SQL and GitHub

### Mohit Kumar
**Role:** AI & ML Developer

- Python and C/C++
- Arduino and IoT
- Embedded Systems
- SQL, NumPy and Pandas

## 🤝 Collaboration

This project is collaboratively developed by Pushpa Kumari and Mohit Kumar using GitHub.

Both developers contribute to improving the application, Machine Learning workflow, and project functionality.

## 📜 Dataset & Acknowledgement

Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains SMS messages labelled as spam or ham and is used for training and evaluating the current classification models.

## ⭐ Support

If you find this project useful, consider giving this repository a star on GitHub!

---

**Developed with Python, Machine Learning and Streamlit.**