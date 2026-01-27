# 📊 Employee Intention to Stay Prediction System

A comprehensive machine learning system for predicting employee retention in the electronics manufacturing sector, developed as part of MRes Artificial Intelligence research at the University of Wolverhampton (2025-2026).

## 🎓 Academic Information

**Author:** Eirika Manandhar  
**Degree:** Master of Research (MRes) in Artificial Intelligence  
**Institution:** University of Wolverhampton  
**Academic Year:** 2025-2026  
**Supervisor:** Prof. Dr Tahir Mahmood

## 📖 Abstract

This research investigates employee retention factors in Vietnam's electronics manufacturing sector using a hybrid Machine Learning and PLS-SEM framework. The system analyzes 604 employees across 29 variables, examining relationships between workplace factors (Perceived Organizational Support, Job Satisfaction, Work-Life Balance) and employee intention to stay.

### Key Findings:
- **Perceived Organizational Support** has the strongest effect on retention (β=0.600)
- **Work-Life Balance** significantly influences intention to stay (β=0.463)
- **Job Satisfaction** showed no significant direct effect
- **Random Forest** achieved 99.80% accuracy in predicting employee retention

## 🚀 Features

- **7 Machine Learning Models:** Random Forest, Gradient Boosting, XGBoost, KNN, Decision Tree, SVM, Logistic Regression
- **Multi-Output Prediction:** Predicts all 5 ITS (Intention to Stay) dimensions simultaneously
- **Advanced Balancing:** Joint-label SMOTEN/RandomOverSampler for handling class imbalance
- **Interactive Dashboards:** Real-time visualizations using Plotly and Seaborn
- **Model Comparison:** Side-by-side comparison with dissertation results
- **Custom Dataset Upload:** Upload your own CSV for predictions
- **Comprehensive Analysis:** Feature importance, SHAP values, model metrics

## 🎯 Model Performance

Results on balanced dataset (10,246 samples after balancing):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.80% | 99.81% | 99.80% | 99.80% |
| Gradient Boosting | 99.75% | 99.75% | 99.75% | 99.75% |
| XGBoost | 99.69% | 99.69% | 99.69% | 99.69% |
| KNN | 99.66% | 99.66% | 99.66% | 99.66% |
| Decision Tree | 99.62% | 99.64% | 99.62% | 99.62% |
| SVM | 94.77% | 94.85% | 94.77% | 94.77% |
| Logistic Regression | 72.27% | 72.43% | 72.27% | 72.13% |

## 📊 Dataset

- **Size:** 604 employees
- **Variables:** 29 (8 demographics + 16 predictors + 5 outcomes)
- **Sectors:** Electronics manufacturing companies in Vietnam
- **Features:**
  - **Demographics (Q1-Q8):** Gender, age, education, tenure, etc.
  - **Perceived Organizational Support (POS1-POS8):** 8 items
  - **Job Satisfaction (JS1-JS5):** 5 items
  - **Work-Life Balance (WLB1-WLB3):** 3 items
  - **Intention to Stay (ITS1-ITS5):** 5 items (target variables)

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **ML Libraries:** Scikit-learn, XGBoost, Imbalanced-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Statistical Analysis:** SciPy, Statsmodels

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

```bash
# Clone the repository
git clone https://github.com/EirikaMK/employee-retention-prediction.git
cd employee-retention-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎮 Usage

### 1. Access the Application
Visit the live app: at `https://employee-retention-prediction-tqburrqiw9uzp7tey2e8tw.streamlit.app/'


### 2. Navigate Through Pages

**🏠 Home:** Overview of the system and research background

**📊 Dashboard:** Interactive visualizations of dataset characteristics

**📈 Model Analysis:** Detailed comparison of all 7 machine learning models

**🔮 Upload & Predict:** Upload your own CSV file for predictions
- Download the sample CSV template
- Ensure your file has columns: Q1-Q8, POS1-POS8, JS1-JS5, WLB1-WLB3, ITS1-ITS5
- Upload and get instant predictions

**📞 Connect With Me:** Contact information and social links
-EmailID: manandhareirika@gmail.com
-LinkedIn: https://www.linkedin.com/in/eirika-manandhar-683211276?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app
-github: https://github.com/EirikaMK/
-portfoli:https://www.eirikamanandhar.com.np/

**About** 
-This dissertation project presents a comprehensive framework combining Machine Learning and Partial Least Squares Structural Equation Modeling (PLS-SEM) to predict and understand employee retention in the Vietnam electronics manufacturing sector.

### 3. Understanding Predictions

The system predicts **Intention to Stay (ITS)** on a 5-point scale:
- **5:** Strongly Agree (High retention)
- **4:** Agree
- **3:** Neutral
- **2:** Disagree
- **1:** Strongly Disagree (High turnover risk)

## 🔬 Methodology

### Data Preprocessing
1. Missing value imputation (median for numerical, mode for categorical)
2. Feature scaling (StandardScaler)
3. Label encoding (1-5 scale converted to 0-4 for ML models)

### Balancing Strategy
- **Joint-label approach:** Combines all 5 ITS columns into single label (e.g., "4-4-4-4-4")
- **SMOTEN/RandomOverSampler:** Balances joint labels
- **Result:** 604 samples → 10,246 balanced samples

### Model Training
- **Multi-output prediction:** All 5 ITS dimensions predicted simultaneously
- **80/20 train-test split** with stratification
- **Cross-validation:** 5-fold CV for robust evaluation
- **Hyperparameter tuning:** Optimized for each model

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (weighted average)
- Confusion matrices
- Feature importance rankings
- SHAP values for interpretability

## 📂 Project Structure

```
employee-retention-prediction/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── setup.py                        # Automates the installation and verification process
├── data/                           # Data directory (not uploaded to GitHub)
│   └── Data POS-ITS 2.csv          # Sample dataset
│
├── models/                         # Models logic
│   ├── analysis_engine.py          # Detailed methodology
│   
└── utils/                          # Helper functions
│    ├── helpers.py
└──images/                          # Helper functions
│   ├── eirika.svg #Images and static files    
└──runtime.txt 

    

```

## 🎯 Key Research Contributions

1. **Hybrid Framework:** Combines ML prediction with PLS-SEM causal analysis
2. **Multi-Output Approach:** Predicts all retention dimensions simultaneously
3. **Joint-Label Balancing:** Novel approach to handle class imbalance
4. **High Accuracy:** Achieved 99.80% with Random Forest
5. **Practical Application:** Deployable system for HR analytics

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{manandhar2026employee,
  author = {Manandhar, Eirika},
  title = {Employee Intention to Stay Prediction Using Hybrid ML-PLS-SEM Framework},
  school = {University of Wolverhampton},
  year = {2026},
  type = {Master's Thesis (MRes Artificial Intelligence)}
}
```

## 📧 Contact

**Eirika Manandhar**
- 📧 Email: manandhareirika@gmail.com
- 🎓 Institution: University of Wolverhampton

## 👨‍🏫 Academic Supervision

**Prof. Dr Tahir Mahmood**  
University of Wolverhampton  
Faculty of Science and Engineering

## 📄 License

This project is part of academic research. For academic or research use, please contact the author.

## 🙏 Acknowledgments

- University of Wolverhampton for academic support
- Prof. Dr Tahir Mahmood for supervision and guidance
- Survey participants from Vietnam's electronics manufacturing sector
- Open-source community for excellent ML libraries


## 🔮 Future Work

- Integration with live HR systems
- Extended features (employee demographics, performance metrics)
- Real-time prediction API
- Mobile application
- Desktop application
- Multi-language support

---

**🎓 Academic Project | University of Wolverhampton | 2025-2026**

*Developed with ❤️ for improving employee retention through data science*
