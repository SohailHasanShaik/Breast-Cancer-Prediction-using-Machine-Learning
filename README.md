# Breast Cancer Prediction using Machine Learning on Wisconsin Dataset

This project demonstrates the application of various machine learning models to predict breast cancer using the Wisconsin Breast Cancer dataset. The models implemented include K-Nearest Neighbors (KNN), Logistic Regression, and Naive Bayes. The project also includes cross-validation to compare model performance.

## Introduction
Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for improving the survival rate of patients. This project aims to build a predictive model using machine learning techniques to assist in the early diagnosis of breast cancer.

## Dataset
The dataset used in this project is the Wisconsin Breast Cancer dataset, which is widely used for binary classification tasks in medical research. The dataset includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing the characteristics of the cell nuclei present in the image.

- **Diagnosis:** Binary target variable indicating whether the cancer is malignant (M) or benign (B).
- **Features:** Various measurements related to the cell nuclei.

## Requirements
To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure
The project contains the following files:
- `main.py`: The main Python script that contains the code for data processing, model training, evaluation, and cross-validation.
- `README.md`: Detailed documentation of the project.
- `dataset_filename.csv`: The dataset file

## Data Preprocessing
The preprocessing steps include:
1. Dropping irrelevant columns (e.g., `ID`).
2. Mapping the diagnosis column to binary values (0 for benign, 1 for malignant).
3. Splitting the dataset into features (`X`) and target (`y`).
4. Splitting the data into training and testing sets with an 80-20 split.

## Modeling
### K-Nearest Neighbors (KNN)
The KNN algorithm is used to classify the data points based on the closest training examples in the feature space. Initially, we use 5 features for prediction, achieving an accuracy of approximately 92%.

### Logistic Regression
Logistic Regression is a linear model for binary classification. By training on all features, we achieve an accuracy of approximately 96%.

### Naive Bayes
The Naive Bayes classifier is based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It achieves an accuracy of approximately 97%.

## Cross-Validation
K-Fold cross-validation is used to evaluate the models' performance more robustly. This approach helps in assessing how the models would generalize to an independent dataset.

## Results
- **K-Nearest Neighbors (KNN):** 92% accuracy, 10-fold CV score of ~88%.
- **Logistic Regression:** 96% accuracy, 10-fold CV score of ~95%.
- **Naive Bayes:** 97% accuracy, 10-fold CV score of ~96%.

## Conclusion
Among the three models, the Naive Bayes classifier showed the highest accuracy for breast cancer prediction. The results suggest that this model could be a reliable tool for early breast cancer diagnosis.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/breast-cancer-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd breast-cancer-prediction
    ```
3. Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the main Python script:
    ```bash
    python main.py
    ```
