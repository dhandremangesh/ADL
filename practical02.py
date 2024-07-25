# Write a Program to implement regularization to prevent the model from overfitting. 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn import metrics

# Load diabetes dataset
diabetes = load_diabetes()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

# Building Lasso Regression Model
lasso = Lasso()

# Fitting model on Train set
lasso.fit(X_train, y_train)

# Calculating Root Mean Squared Error.
train_rmse_lasso = np.sqrt(metrics.mean_squared_error(y_train, lasso.predict(X_train)))
test_rmse_lasso = np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_test)))
print("Lasso Train RMSE:", np.round(train_rmse_lasso, 5))
print("Lasso Test RMSE:", np.round(test_rmse_lasso, 5))

# Building Ridge Regression Model
ridge = Ridge()

# Fitting model on Train set
ridge.fit(X_train, y_train)

# Calculating Root Mean Squared Error.
train_rmse_ridge = np.sqrt(metrics.mean_squared_error(y_train, ridge.predict(X_train)))
test_rmse_ridge = np.sqrt(metrics.mean_squared_error(y_test, ridge.predict(X_test)))
print("Ridge Train RMSE:", np.round(train_rmse_ridge, 5))
print("Ridge Test RMSE:", np.round(test_rmse_ridge, 5))