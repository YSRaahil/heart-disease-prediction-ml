
🫀 Heart Attack Detection using Machine Learning

A data-driven prediction system that estimates the likelihood of a heart attack using patient health data and machine learning algorithms.
The model achieves up to 89% accuracy, making it a strong diagnostic support tool for healthcare analytics and preventive medicine.


🚀 Overview

This project develops a Heart Attack Risk Prediction Model using supervised learning techniques.
It processes 13 key medical parameters and outputs whether an individual is likely to experience a heart attack.

Highlights:

🧮 Trained on 500 patient records

⚙️ Evaluated using 4 ML models (Logistic Regression, Random Forest, SVM, KNN)

📊 Achieved 89% accuracy and 91% recall

💾 Lightweight: model size < 1 MB

⏱️ Average inference time: 0.02 seconds per sample

🧠 Dataset

Attribute	Description

Age	Age of the individual
Sex	Gender (1 = Male, 0 = Female)
Chest Pain Type (cp)	0–3 (4 categories)
Resting BP (trtbps)	in mmHg
Cholesterol (chol)	Serum cholesterol (mg/dl)
Fasting Blood Sugar (fbs)	>120 mg/dl (1 = True, 0 = False)
Rest ECG (restecg)	Electrocardiographic results (0–2)
Max Heart Rate (thalachh)	Maximum heart rate achieved
Exercise Induced Angina (exng)	1 = Yes, 0 = No
Oldpeak	ST depression value
Slope (slp)	Slope of ST segment
CA (caa)	No. of major vessels (0–3)
Thalassemia (thall)	1–3 (normal, fixed defect, reversible defect)


> 📘 Dataset Source: Heart Disease UCI Dataset – Kaggle
🧩 Total Samples: 500 | Features: 13 | Classes: Binary (0 = No attack, 1 = Risk of attack)


⚙️ Workflow

1. Data Preprocessing

Removed duplicates & missing values

Scaled numerical features using StandardScaler

Split dataset: 80% training / 20% testing


2. Model Training

Algorithms used:

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors

Best performing model: Logistic Regression 

🛠️ Tech Stack

Category	Tools

Language	Python 3.x
Libraries	pandas, numpy, scikit-learn, matplotlib, seaborn, pickle
Environment	Jupyter Notebook / VS Code
Runtime	CPU-based (Intel i5 tested)

📂 Project Structure

heart-attack-detection/
│
├── model/
│   └── heart_attack_model.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── data/
│   └── heart.csv
│
├── requirements.txt
└── README.md


🧪 How to Run

1. Clone the repository

git clone https://github.com/your-username/heart-attack-detection.git
cd heart-attack-detection


2. Install dependencies

pip install -r requirements.txt


3. Train the model

python src/train.py


4. Make predictions

python src/predict.py


📈 Key Results

✅ Accuracy: 0.89

✅ Precision: 0.87

✅ Recall: 0.91

✅ F1-Score: 0.89

✅ ROC-AUC: 0.92


> 🧠 The model demonstrated strong generalization with minimal overfitting and stable performance across test splits.

🔍 Insights

Cholesterol, age, and chest pain type are top 3 contributing factors.

Exercise-induced angina and oldpeak are strong risk indicators.

Feature importance graph available in train.py output.

🧾 License

Licensed under the MIT License – free to use, modify, and distribute with attribution.

🙌 Acknowledgments

UCI Machine Learning Repository for dataset reference

Kaggle community for open-access data

Scikit-learn and Python developers for open-source tools
