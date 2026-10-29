# CreditWise AI – Data-Driven Loan Approval Prediction

This is a machine learning project where I explored applicant information and trained classification models to predict whether a loan would be approved.

## What I did

- Checked the dataset and handled missing values using mean and most-frequent imputation.
- Explored loan approval, applicant income, credit score, and other features using charts and a correlation heatmap.
- Encoded categorical features and removed `Applicant_ID`.
- Split the data into training and testing sets and applied feature scaling.
- Trained and compared Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.
- Tried feature engineering by creating squared features for `DTI_Ratio` and `Credit_Score`, then evaluated the models again.

## Model Results

The table below shows the results **after feature engineering** on the test set.

| Model | Precision | Recall | Accuracy | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 78.46% | 83.61% | **88.00%** | **80.95%** |
| K-Nearest Neighbors (KNN) | 64.58% | 50.82% | 76.50% | 56.88% |
| Gaussian Naive Bayes | **81.13%** | 70.49% | 86.00% | 75.44% |

**Gaussian Naive Bayes achieved the highest precision (81.13%)** among the three models. Logistic Regression achieved the highest accuracy (88%), recall (83.61%), and F1-score (80.95%). So, the better model depends on which evaluation metric matters most.

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Files

- `Loan_Approval_Prediction.ipynb` – notebook containing the data analysis, preprocessing, model training, and evaluation.
- `loan_approval_data.csv` – dataset used by the notebook.

## How to Run

Keep the notebook and `loan_approval_data.csv` in the same folder. Install the required Python libraries, then open the notebook in Jupyter Notebook or Google Colab and run the cells. The CSV filename must match the filename used in the notebook.
