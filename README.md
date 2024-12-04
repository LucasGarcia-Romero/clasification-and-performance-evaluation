# Machine Learning Project: Classification and Performance Evaluation

## Overview

This project demonstrates the use of machine learning algorithms for classification tasks, specifically focusing on Gaussian Naive Bayes and Support Vector Machines (SVM). The project includes data preprocessing, model training, evaluation using cross-validation, and statistical tests to compare model performance.

## Contents

- **Data Reading**: The project reads training and testing datasets from TSV files and separates features and labels for model training and evaluation.
- **Data Normalization**: Features are normalized using StandardScaler to improve model performance.
- **Model Implementation**: 
  - Gaussian Naive Bayes and SVM models are implemented for classification tasks.
- **Performance Evaluation**:
  - Accuracy scores are computed to evaluate model performance.
  - Cross-validation is used to assess the stability of the models.
  - The McNemar test is applied to compare the performance of different models statistically.
  
## Requirements

Make sure to have the following libraries installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## How to Run

1. Clone the repository or download the script.
2. Place the `TP1_train.tsv` and `TP1_test.tsv` datasets in the same directory as the script.
3. Run the script in your preferred Python environment. For example, in Jupyter Notebook or Google Colab.

## Functions

- **McNemarTest(Yt, predC1, predC2)**: Performs the McNemar test to compare the performance of two classifiers on the same dataset.

## Outputs

The project outputs the accuracy scores of the trained models and results of the McNemar test, allowing users to compare the effectiveness of different classification algorithms.

## Conclusion

This project serves as an introduction to machine learning classification techniques and performance evaluation methods. It provides a framework for experimenting with different models and understanding their strengths and weaknesses in classification tasks.
