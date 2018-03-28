# https://github.com/juliensimon/aws/blob/master/ML/scikit/linearregression.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y_train = np.array([1, 2.5, 2, 3, 5, 4, 5.5, 7, 6, 8])

# Transform vectors into 10-line, 1-column matrices
X_train = X_train.reshape((-1, 1))
Y_train = Y_train.reshape((-1, 1))

# Create linear regression objects
regr_lin = linear_model.LinearRegression()
regr_sgd = linear_model.SGDRegressor()

# Train the models
regr_lin.fit(X_train, Y_train)
regr_sgd.fit(X_train, Y_train)

# Save model for future use
from sklearn.externals import joblib
joblib.dump(regr_lin, 'linearregressionmodel.pkl')
joblib.dump(regr_sgd, 'sgdregressionmodel.pkl')

# Make predictions
Y_pred_lin = regr_lin.predict(X_train)
Y_pred_sgd = regr_sgd.predict(X_train)

# Print variance and RMSE
print("Linear regression")
print("- Coefficient: %s " % regr_lin.coef_)
print("- Intercept: %s " % regr_lin.intercept_)
print("- Variance: %s " % r2_score(Y_train, Y_pred_lin))
print("- RMSE: %s " % mean_squared_error(Y_train, Y_pred_lin))
print("")
print("SGD regression")
print("- Coefficient: %s " % regr_sgd.coef_)
print("- Intercept: %s " % regr_sgd.intercept_)
print("- Variance: %s " % r2_score(Y_train, Y_pred_sgd))
print("- RMSE: %s " % mean_squared_error(Y_train, Y_pred_sgd))

# Plot outputs
plt.scatter(X_train, Y_train,  color='blue')

plt.scatter(X_train, Y_pred_lin, color='orange')
plt.plot(X_train, Y_pred_lin, color='orange', linewidth=1)

plt.scatter(X_train, Y_pred_sgd, color='yellow')
plt.plot(X_train, Y_pred_sgd, color='orange', linewidth=1)

plt.show()
