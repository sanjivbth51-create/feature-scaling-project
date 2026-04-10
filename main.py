# Task 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Task 2: Create dataset
data = {
    "feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "feature_2": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    "target":    [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Task 3: Separate features (X) and target (y)
X = df[["feature_1", "feature_2"]]
y = df["target"]

# Task 4: Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Task 5: Apply StandardScaler correctly
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler
X_test_scaled = scaler.transform(X_test)

# Task 6: Print results
print("Scaled Training Data:")
print(X_train_scaled)

print("\nScaled Test Data:")
print(X_test_scaled)