# Credit-Card-Fraud-Detection-on-Imbalanced-Data-Using-Logistic-Regression

This project focuses on building a logistic regression model to detect fraudulent credit card transactions in an imbalanced dataset. It demonstrates how to preprocess sensitive and skewed financial data and apply basic resampling techniques to improve model performance.

### Project Overview

* **Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* **Goal:** Detect fraudulent transactions using a binary classification model.
* **Challenge:** The dataset is highly imbalanced (fraudulent transactions are extremely rare as we have 284315 legit normal transactions and only 492 fraudulent transactions.

### Key Features

* Data cleaning and exploratory analysis
* Handling missing values
* Addressing class imbalance using undersampling
* Building and evaluating a logistic regression model
* Measuring performance on training and test sets

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter/Colab

### Workflow

1. **Data Loading and Exploration**

   * Load the dataset
   * Inspect structure, types, and missing values
   * Understand the class distribution (legit vs fraud)

2. **Handling Imbalanced Data**

   * Apply undersampling to create a balanced dataset
   * Merge a subset of legit transactions with all fraud transactions

3. **Preprocessing**

   * Drop missing values
   * Convert `Amount` column to float
   * Separate features and labels

4. **Modeling**

   * Split data using `train_test_split` with stratification
   * Train a logistic regression model
   * Evaluate model on both training and test data

5. **Evaluation Metrics**

   * Accuracy on training and testing sets
   * (Optional) Additional metrics like precision, recall, or confusion matrix can be added for deeper insight
   * 94 percent training accuracy
   * 91 percent test accuracy

### How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies (optional if using Colab):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or Python script:

   * In Jupyter or Colab, open and execute cells step by step
   * Or run the Python script directly (ensure the dataset path is correct)

### Notes

* The dataset is anonymized and includes PCA-transformed features for privacy.
* Accuracy alone may not be sufficient for evaluating fraud detection systems. Future improvements could include using SMOTE, ensemble methods, or deep learning models.

### License

This project is for educational purposes. Refer to the original dataset:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


