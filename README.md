# Credit Card Fraud Detection

Welcome to the Credit Card Fraud Detection project! This repository contains code and resources for detecting fraudulent credit card transactions using various machine learning models. The dataset used for this project is sourced from Kaggle, and comprehensive exploratory data analysis (EDA) has been performed to gain insights into the data. Several machine learning models have been implemented, including Logistic Regression and Support Vector Classifier (SVC), with performance evaluated based on accuracy, precision, and other metrics.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a significant issue that causes substantial financial losses. The goal of this project is to build a reliable model to detect fraudulent transactions and minimize these losses. By leveraging machine learning techniques, we can identify patterns and anomalies in transaction data that indicate potential fraud.

## Dataset

The dataset used for this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) available on Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

## Exploratory Data Analysis (EDA)

In the EDA section, various analyses are performed to understand the structure and characteristics of the data, including:
- Distribution of features
- Correlation between features
- Handling imbalanced data
- Feature scaling

Visualizations and statistical summaries are used to highlight key insights and prepare the data for model building.

## Machine Learning Models

The following machine learning models have been implemented to detect credit card fraud:
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Isolation Forest
- One-Class SVM
- Naive Bayes (GaussianNB)
- Gradient Boosting Classifier
- Local Outlier Factor

Each model is trained and tested using appropriate techniques to ensure reliable performance.

## Model Evaluation

The performance of the models is evaluated using various metrics, including:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

These metrics help assess the effectiveness of the models in detecting fraudulent transactions.

## Installation

To run the code in this repository, you need to have Python installed. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Follow these steps to run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter notebook `Credit_Card_Fraud_Detection.ipynb` and run the cells to explore the data, build models, and evaluate their performance.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
