# ------------------------------------------------------------
# Customer Churn Prediction using Feedforward Neural Network
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
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1200, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]
], dtype=float)


y = np.array([
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 1
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
# Step 8: Predict New Customer
# ------------------------------------------------------------

new_customer = np.array([
    [46, 1450, 5, 6, 9]
], dtype=float)


new_customer_scaled = scaler.transform(new_customer)


prediction = model.predict(new_customer_scaled, verbose=0)


# ------------------------------------------------------------
# Step 9: Display Prediction
# ------------------------------------------------------------

if prediction[0][0] >= 0.5:
    print("\nPrediction: Customer may leave")
else:
    print("\nPrediction: Customer may stay")


print("Prediction Probability:", prediction[0][0])