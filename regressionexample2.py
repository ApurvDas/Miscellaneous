import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Create synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a standard linear regression model
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Train a Lasso regression model
lasso_model = Lasso(alpha=0.1)  # Adjust alpha for regularization strength
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Train a Ridge regression model
ridge_model = Ridge(alpha=0.1)  # Adjust alpha for regularization strength
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Calculate Mean Squared Error for each model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Print results
print(f'MSE - Linear Regression: {mse_linear}')
print(f'MSE - Lasso Regression: {mse_lasso}')
print(f'MSE - Ridge Regression: {mse_ridge}')

# Plot coefficients to compare
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(range(len(linear_model.coef_)), linear_model.coef_)
plt.title('Linear Regression Coefficients')

plt.subplot(1, 3, 2)
plt.bar(range(len(lasso_model.coef_)), lasso_model.coef_)
plt.title('Lasso Regression Coefficients')

plt.subplot(1, 3, 3)
plt.bar(range(len(ridge_model.coef_)), ridge_model.coef_)
plt.title('Ridge Regression Coefficients')

plt.tight_layout()
plt.show()
