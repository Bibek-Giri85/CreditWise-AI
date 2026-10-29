# CreditWise Loan System

This is a machine learning project where I explored loan application data and compared a few classification models to predict whether a loan would be approved.

## What I did

- Checked the data and handled missing values.
- Used charts to explore loan approval, applicant income, credit score, and other features.
- Encoded categorical columns and removed the applicant ID.
- Split the data into training and testing sets and scaled the features.
- Trained and evaluated Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.
- Tried feature engineering using squared values of DTI ratio and credit score, then compared the models again.

## Results

I evaluated the models using accuracy, precision, recall, F1-score, and a confusion matrix. After feature engineering, Logistic Regression achieved **88% accuracy** and an **F1-score of about 0.81** on the test set.

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Files

- `Loan_Approval_Prediction.ipynb` – notebook containing the data analysis, preprocessing, model training, and evaluation.
- `loan_approval_data.csv` – dataset used by the notebook.

## How to run

Keep the notebook and `loan_approval_data.csv` in the same folder. Install the required libraries, then open and run the notebook in Jupyter Notebook or Google Colab. The CSV filename should match the name used in the code.
