# ------------------------------------------------------------
# Loan Approval Prediction using Feedforward Neural Network
# ------------------------------------------------------------

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ------------------------------------------------------------
# Step 1: Create Dataset
# ------------------------------------------------------------

X = np.array([
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 8000, 1],
    [60000, 750, 500000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 800, 700000, 10000, 1],
    [35000, 650, 250000, 9000, 1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 0],
    [70000, 780, 600000, 10000, 1]
], dtype=float)


y = np.array([
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1
])


# ------------------------------------------------------------
# Step 2: Split Dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# Step 3: Standardize Features
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# Step 4: Create FNN Model
# ------------------------------------------------------------

model = Sequential([
    
    Dense(8, activation="relu", input_shape=(5,)),
    
    Dense(4, activation="relu"),
    
    Dense(1, activation="sigmoid")
])


# ------------------------------------------------------------
# Step 5: Compile Model
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Step 6: Train Model
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=2,
    verbose=1
)


# ------------------------------------------------------------
# Step 7: Evaluate Model
# ------------------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Accuracy:", accuracy * 100, "%")


# ------------------------------------------------------------
# Step 8: Predict New Applicant
# ------------------------------------------------------------

new_applicant = np.array([
    [55000, 720, 400000, 10000, 1]
], dtype=float)


new_applicant_scaled = scaler.transform(new_applicant)


prediction = model.predict(
    new_applicant_scaled,
    verbose=0
)


# ------------------------------------------------------------
# Step 9: Display Prediction
# ------------------------------------------------------------

if prediction[0][0] >= 0.5:
    print("\nPrediction: Loan may be approved")
else:
    print("\nPrediction: Loan may be rejected")


print("Prediction Probability:", prediction[0][0])