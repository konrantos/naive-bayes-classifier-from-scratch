# Naive Bayes Classifier — From Scratch

This project implements a full Naive Bayes Classifier (NBC) **from scratch** in Python, without using external libraries such as `scikit-learn`. The implementation is designed to demonstrate how NBC works under the hood, including both **discrete** and **continuous** features.

## Key Features

- **Manual Naive Bayes Implementation**
  - No use of scikit-learn — pure Python
  - Supports both discrete and continuous features
  - Calculates prior and conditional probabilities
  - Laplace smoothing for discrete features
  - Gaussian probability estimation for continuous features

- **Evaluation**
  - 70/30 train-test split
  - Computes classification accuracy
  - Reports misclassified instances

- **User Prediction**
  - Accepts user-defined sample for prediction
  - Returns posterior probabilities per class

## Dataset Format

- CSV file must contain attribute-type pairs in the first row:
  - `C` = continuous
  - `D` = discrete
- Last column must be the class label

Example:
```
"sepal.length",C,"sepal.width",C,"petal.length",C,"petal.width",C,"variety"
5.1,3.5,1.4,.2,"Setosa"
```

## How to Run

```bash
git clone https://github.com/konrantos/naive-bayes-classifier-from-scratch.git
cd naive-bayes-classifier-from-scratch
python naive_bayes_from_scratch.py
```

> Make sure the CSV dataset file is in the same folder. The script will prompt you for the file name.

## Output Example

- Prior probabilities for each class
- Conditional probabilities or Gaussian parameters per feature
- Classification accuracy and misclassified instances
- Posterior probabilities for custom input

## Sample Dataset

- You may test the classifier using any labeled dataset like `iris.csv`, as long as it includes attribute-type headers.

## Requirements

- Python 3.11+
- No third-party libraries required

## License

MIT License
