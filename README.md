
# Credit Card Fraud Detection

## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Environment & Dependencies](#environment--dependencies)  
- [Project Structure](#project-structure)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)  
- [Time‑Aware SMOTE Resampling](#time‑aware-smote-resampling)  
- [Modeling](#modeling)  
  - [1. Logistic Regression](#1-logistic-regression)  
  - [2. Random Forest](#2-random-forest)  
  - [3. XGBoost](#3-xgboost)  
  - [4. Autoencoder + XGBoost Hybrid](#4-autoencoder--xgboost-hybrid)  
- [Evaluation & Results](#evaluation--results)  
- [Feature Importance](#feature-importance)  
- [Usage](#usage)  
- [References](#references)  
- [License](#license)  

---

## Project Overview  
This repository implements a comprehensive pipeline for detecting credit card fraud using the popular “Credit Card Fraud Detection” dataset. It covers:  
1. **Exploratory Data Analysis** to understand class imbalance and feature distributions  
2. **Preprocessing** including scaling of the `Time` and `Amount` features  
3. **Time‑Aware SMOTE** to handle severe class imbalance while preserving temporal ordering  
4. **Comparative Modeling** with classical and ensemble classifiers  
5. **A Hybrid Approach** using an Autoencoder to extract features fed into XGBoost  
6. **Evaluation** via confusion matrices and ROC curves  
7. **Feature Importance Analysis** to interpret model predictions  

---

## Dataset  
- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Records: 284,807 transactions made by European cardholders in September 2013  
- Features:  
  - `Time`: Seconds elapsed between each transaction and the first transaction  
  - `V1`–`V28`: PCA‑transformed anonymized numerical features  
  - `Amount`: Transaction amount  
  - `Class`: Target variable (0 = non‑fraud, 1 = fraud)  

---

## Environment & Dependencies  
1. **Python 3.8+**  
2. Key libraries:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap tensorflow

---

## Project Structure

```
.
├── Credit_Card_Fraud_Detection.ipynb   ← Jupyter notebook with full pipeline  
├── requirements.txt                    ← List of packages & versions  
├── figures/                            ← Saved plots (histograms, heatmaps, ROC curves)  
└── README.md                           ← This file  
```

---

## Exploratory Data Analysis (EDA)

1. **Class Imbalance**

   * Only \~0.17% of transactions are fraud → critical need for resampling.
2. **Feature Distributions**

   * Histograms of PCA components and `Amount` & `Time` by class.
3. **Correlation Analysis**

   * Heatmap of features to check for multicollinearity.

---

## Preprocessing & Feature Engineering

1. **Scaling**

   * `Time` and `Amount` scaled via `StandardScaler` (or `MinMaxScaler` for autoencoder).
2. **Sorting**

   * Transactions sorted chronologically to simulate real‑time flow.
3. **Train/Test Split**

   * Stratified split to maintain class ratio.

---

## Time‑Aware SMOTE Resampling

* **Purpose:** Handle severe class imbalance without breaking temporal dependence.
* **Procedure:**

  1. Bin the training set into 10 equal‐width time intervals.
  2. Within each bin, if fraud samples exist, apply SMOTE to balance classes.
  3. Reassemble the bins to form the final training set.

---

## Modeling

### 1. Logistic Regression

* Baseline linear classifier.
* Metrics: ROC AUC, confusion matrix.

### 2. Random Forest

* Ensemble of decision trees.
* Useful for capturing non‐linear patterns.

### 3. XGBoost

* Gradient‑boosted trees with regularization.
* Strong performance on tabular data.

### 4. Autoencoder + XGBoost Hybrid

* **Autoencoder**

  * Architecture: Input → 14 → 7 → 14 → Output
  * Trained on normalized features to learn compressed representations.
* **Hybrid Pipeline**

  1. Train AE; extract 7‐dim bottleneck features.
  2. Concatenate bottleneck features with original features.
  3. Train XGBoost on this enriched feature set.

---

## Evaluation & Results

* **Confusion Matrix** for each model (saved under `figures/`).
* **Combined ROC Curve** overlaying all four approaches for direct AUC comparison.
* Summary of AUC scores printed in notebook output.

---

## Feature Importance

* **Tree‑based Models** (RF, XGBoost): Plotted via `plot_importance()` (gain‐based).
* **Logistic Regression**: Absolute coefficient values.
* **Hybrid AE+XGB**: Top‑10 features by gain in the combined feature set.

---

## Usage

1. Clone this repo:

   ```bash
   git clone https://github.com/<your‑username>/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the notebook:

   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```
4. Inspect outputs in `figures/` or modify parameters as needed.

---

## References

* Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2014). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. *IEEE Transactions on Neural Networks and Learning Systems*.
* Kaggle Dataset: [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
