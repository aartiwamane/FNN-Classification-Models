# Feedforward Neural Network Classification Models

This repository contains two binary classification models developed using
Feedforward Neural Networks (FNN).

The models are implemented using Python, TensorFlow/Keras, NumPy, and
Scikit-learn.

## Projects

### 1. Customer Churn Prediction

Predicts whether a customer is likely to stay with or leave a service.

Features:

- Age
- Monthly Charges
- Tenure
- Number of Complaints
- Customer Support Calls

Output:

- 0 → Customer will stay
- 1 → Customer will leave

Project folder:

`Customer_Churn_FNN/`

---

### 2. Loan Approval Prediction

Predicts whether a loan application is likely to be approved or rejected.

Features:

- Applicant Income
- Credit Score
- Loan Amount
- Existing EMI
- Employment Status

Output:

- 0 → Loan rejected
- 1 → Loan approved

Project folder:

`Loan_Approval_FNN/`

---

## Neural Network Architecture

Both models use a Feedforward Neural Network with:

- Input layer
- Dense hidden layer with ReLU activation
- Dense hidden layer with ReLU activation
- Output layer with Sigmoid activation

Architecture:

Input Features
↓
Dense Layer (8 neurons, ReLU)
↓
Dense Layer (4 neurons, ReLU)
↓
Output Layer (1 neuron, Sigmoid)

---

## Machine Learning Workflow

The projects follow these steps:

1. Create the dataset
2. Split the dataset into training and testing sets
3. Standardize numerical features
4. Create the FNN model
5. Compile the model
6. Train the model
7. Evaluate model accuracy
8. Predict the result for a new input

---

## Technologies Used

- Python
- NumPy
- Scikit-learn
- TensorFlow
- Keras

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FNN-Classification-Models.git
