# 💰 Loan Approval Predictor

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://loanapproval-project.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.2-FFD700)](https://catboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange?logo=scikit-learn)](https://scikit-learn.org/)

---

## 🚀 Live Demo

👉 **Try it here:** [Loan Approval App](https://loanapproval-project.streamlit.app/)

---

## 📌 Overview

This project predicts whether a **loan application will be approved or rejected** using machine learning.  
It leverages **CatBoost** for highly accurate classification and features an **interactive Streamlit web app** for real-time predictions.

Users can input applicant details such as personal information, employment status, income, credit history, and loan requirements to get instant approval predictions with **99.65% test accuracy**.

---

## ⚙️ Features

✅ Real-time loan approval predictions with 99.65% accuracy  
✅ Interactive web interface with 4 comprehensive sections  
✅ Exploratory Data Analysis with 15+ visualizations  
✅ 33 features analyzed per application  
✅ Class imbalance handled using SMOTE  
✅ Professional presentation with model insights  

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|--------|
| **Frontend** | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), Custom CSS |
| **Backend / ML** | [Python](https://www.python.org/), [CatBoost](https://catboost.ai/), [Scikit-learn](https://scikit-learn.org/) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Model Persistence** | [Joblib](https://joblib.readthedocs.io/) |
| **Class Balancing** | [Imbalanced-learn](https://imbalanced-learn.org/) (SMOTE) |

---

## 📂 Project Structure

```
loan-approval-predictor/
│
├── Loan_Approval_Project.ipynb   # Data exploration, preprocessing, model training
├── cleaned_df.csv                # Cleaned dataset (20,000 records)
├── CatBoost.pkl                  # Trained model (2.7 MB)
├── loan_approval_app.py          # Streamlit application
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
└── LICENSE                       # MIT License
```

---

## 🧠 Model Overview

- **Algorithm:** CatBoost Classifier with Pipeline  
- **Target Variable:** Loan Approved (1) / Rejected (0)  
- **Training Data:** 16,000 samples (80%)  
- **Test Data:** 4,000 samples (20%)  
- **Features:** 33 comprehensive attributes  

### 🔧 Training Workflow

1. Data cleaning & preprocessing  
2. Feature engineering & encoding  
3. Handling missing values & outliers  
4. Class balancing with SMOTE  
5. Training CatBoost pipeline  
6. Model evaluation & validation  
7. Saving trained model as `CatBoost.pkl`  

---

## 🖥️ How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/zeyadmedhat/loan-approval-predictor.git
   cd loan-approval-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run loan_approval_app.py
   ```

4. Visit the local URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 📦 Requirements

From `requirements.txt`:

```
streamlit==1.29.0
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
plotly==5.18.0
scikit-learn==1.3.2
catboost==1.2.2
imbalanced-learn==0.11.0
```

---

## 💻 Application Pages

### 🏠 Home
- Project overview and key statistics
- How the system works (3-step process)
- Why choose this system
- Quick navigation guide

### 📊 EDA (Exploratory Data Analysis)
- **Dataset Overview:** Summary statistics and distributions
- **Credit Analysis:** Education, marital status, home ownership impacts  
- **Financial Patterns:** Income, assets, debt analysis  
- **Target Distribution:** Demographics and historical factors  

### 🔮 Prediction
- Input 33 applicant features across 4 organized tabs
- Get instant approval/rejection prediction
- View probability gauge and application summary

### 📑 Presentation
- **Project Overview:** Problem statement and solution
- **Model Details:** Architecture, pipeline, algorithms
- **Performance Metrics:** Test results, confusion matrix, insights
- **Deployment Info:** Tech stack and implementation details

---

## 🔍 Features Analyzed (33 Total)

**Personal (4):** Age, Education, Marital Status, Dependents  
**Employment (3):** Status, Experience, Job Tenure  
**Financial (7):** Income, Assets, Liabilities, Net Worth, Savings, Checking, Monthly Income  
**Credit (8):** Score, Utilization, Open Lines, Inquiries, History Length, Payment History, Utility Bills, Total DTI  
**Loan (7):** Amount, Duration, Purpose, Base Rate, Interest Rate, Monthly Payment, DTI  
**History (4):** Monthly Debt, Bankruptcy, Previous Defaults, Home Ownership  

---

## 🛠️ Troubleshooting

**Port already in use:**
```bash
streamlit run loan_approval_app.py --server.port 8502
```

**Module not found:**
```bash
pip install -r requirements.txt
```

**File not found:** Ensure `CatBoost.pkl` and `cleaned_df.csv` are in the same directory

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the **MIT License** — you're free to use, modify, and distribute it for educational or personal purposes. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Zeyad Medhat**  
Data Scientist | Machine Learning Engineer

💼 [LinkedIn](https://linkedin.com/in/zeyad-medhat) | 💻 [GitHub](https://github.com/zeyadmedhat) | 📧 zeyadmedhat.official@gmail.com

---

<div align="center">

**Built with ❤️ by Zeyad Medhat | © 2025**

</div>
