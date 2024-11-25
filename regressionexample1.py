#Applied Machine Learning: Linear Regression and Logistic Regression
##pip install numpy pandas matplotlib statsmodels scikit-learn
import numpy as np
import pandas as pd
import statsmodels.api as sm


#Example1: Multiple Linear Regression
# Sample Data
data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [5, 4, 3, 2, 1],
    'Y': [10, 9, 8, 7, 6]
}
df = pd.DataFrame(data)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Define the independent variables and dependent variable
X = df[['X1', 'X2']]
Y = df['Y']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(Y, X).fit()

# Print the model summary
print("\nModel Summary:")
print(model.summary())

# Print parameters (coefficients)
print("\nCoefficients:")
print(model.params)

# Print predicted values
predicted_values = model.predict(X)
print("\nPredicted Values:")
print(predicted_values)

# Print constants
print("\nModel Constants:")
print("Intercept (Beta0):", model.params[0])
print("Slope (Beta1):", model.params[1])
print("Slope (Beta2):", model.params[2])


###########
#Example2: Logistic Regression

# Sample Data
data = {
    'X1': [0, 1, 2, 3, 4, 5],
    'X2': [5, 4, 3, 2, 1, 0],
    'Y': [0, 0, 0, 1, 1, 1]  # Binary outcome
}
df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['X1', 'X2']]
Y = df['Y']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the model
model = sm.Logit(Y, X).fit()

# Print the summary
print(model.summary())


# Continue from the previous example

# Predicting probabilities
predicted_probabilities = model.predict(X)
print("\nPredicted Probabilities:")
print(predicted_probabilities)

# Class predictions based on a threshold (0.5)
predicted_classes = (predicted_probabilities >= 0.5).astype(int)
print("\nPredicted Classes:")
print(predicted_classes)


############################################3

#Example 3: Logistic regression with sklearn
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Sample Data
data = {
    'X1': [0, 1, 2, 3, 4, 5],
    'X2': [5, 4, 3, 2, 1, 0],
    'Y': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['X1', 'X2']]
Y = df['Y']

# Create and fit the model
model = LogisticRegression()
model.fit(X, Y)

# Print coefficients
print("\nCoefficients:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions
predicted_classes = model.predict(X)
print("\nPredicted Classes using scikit-learn:")
print(predicted_classes)

# Confusion Matrix
cm = confusion_matrix(Y, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
report = classification_report(Y, predicted_classes)
print("\nClassification Report:")
print(report)
#########################################################
#Additional analysis for above logistic regression problem
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Continue from the previous example

# Cross-validation
scores = cross_val_score(model, X, Y, cv=5)  # 5-fold cross-validation
print("\nCross-Validation Scores:")
print(scores)
print("Mean Cross-Validation Score:", scores.mean())


# Create a new range for X1
X1_range = np.linspace(-1, 6, 100)
X2_value = 2  # Fixing X2 to a specific value for visualization

# Predict probabilities across the range
X_new = pd.DataFrame({'X1': X1_range, 'X2': X2_value})
X_new = sm.add_constant(X_new)
probabilities = model.predict(X_new)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X1_range, probabilities, label='Logistic Regression Curve')
plt.scatter(df['X1'], df['Y'], color='red', label='Data Points')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold (0.5)')
plt.title('Logistic Regression Curve')
plt.xlabel('X1')
plt.ylabel('Probability of Y=1')
plt.legend()
plt.show()

###############################################
#Example 4: More complete example of linear regression using sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Read input Dataset
data = pd.read_csv('your_data.csv')

X = data[['feature1', 'feature2', 'feature3']]  # independent variables
y = data['target_variable']                       # dependent variable

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define Model
model = LinearRegression()

#Fit Model/Solve the model for given dataset
model.fit(X_train, y_train)

#Do Predictions
y_pred = model.predict(X_test)

#Evaluation for Model Performance:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Review solved model parameters
coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

################################################################################
