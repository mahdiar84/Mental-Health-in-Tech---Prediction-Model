# 🧠 Mental Health in Tech - Prediction Model

This project predicts whether a person working in the tech industry is likely to seek mental health treatment based on various personal, professional, and behavioral features collected in a survey.

---

## 📊 Dataset

- **Source:** [Kaggle - OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Target Variable:** `treatment` (Yes/No)
- **Features Include:**
  - Age, gender
  - Remote work, benefits
  - Family history of mental illness
  - Company size and wellness programs
  - etc.

---

## ⚙️ Tools & Technologies

- **Python Core**
- **pandas**, **numpy** – for data handling
- **matplotlib**, **seaborn** – for visualization
- **scikit-learn** – for machine learning models and evaluation

---

## 🛠️ Workflow

1. **Data Cleaning**
   - Removed irrelevant columns (comments, state, etc.)
   - Handled missing values (categorical: mode, numerical: median)

2. **Preprocessing**
   - Converted categorical variables using One-Hot Encoding
   - Scaled numerical features using `StandardScaler`

3. **Modeling**
   - Compared three models:
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Support Vector Machine (SVM)

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix (Visualized)
   - Classification Report
   - Accuracy Comparison Bar Chart

---

## 📈 Results

The models were evaluated using a 70/30 train-test split with stratification for balanced classes. Model performance was visualized using seaborn plots and saved locally.

---

## 🧠 Outcome

This project aims to raise awareness about mental health in the tech industry while demonstrating a practical machine learning pipeline from data cleaning to model comparison.

---

## 📌 Note

This project was initially written from scratch, and later refined with the help of AI to enhance clarity, performance, and structure.

---

## ✅ Author
Mahdiar
Mahdiar — Student & AI/Machine Learning Enthusiast  
Project part of the **ARCH Phase 1 Roadmap**
