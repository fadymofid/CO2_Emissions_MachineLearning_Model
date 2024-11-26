import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import shuffle

data = pd.read_csv("co2_emissions_data.csv")

print("Missing values:\n", data.isnull().sum())
pd.set_option('display.max_columns', None)
numeric_features = data.select_dtypes(include=['float64', 'int64'])
print("Summary statistics:\n", numeric_features.describe())

pairplot = sns.pairplot(data, diag_kind='hist')
pairplot.savefig('pairplot.png')
plt.show()


X = data.drop(columns=["CO2 Emissions(g/km)", "Emission Class"])
y = data["CO2 Emissions(g/km)"]
numeric_features = X.select_dtypes(include=['float64', 'int64'])

categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    data[col] = LabelEncoder().fit_transform(data[col])

correlation_matrix = data.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig('heatmap.png')
plt.show()

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for col in numeric_features.columns:
    mean = X_train[col].mean()
    range_ = X_train[col].max() - X_train[col].min()
    X_train[col] = (X_train[col] - mean) / range_
    X_test[col] = (X_test[col] - mean) / range_

strong_features = correlation_matrix["CO2 Emissions(g/km)"].abs().sort_values(ascending=False).index[1:3]
X_train_lr = X_train[strong_features].values
X_test_lr = X_test[strong_features].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[strong_features[0]], X[strong_features[1]], y, alpha=0.5)
ax.set_xlabel(strong_features[0])
ax.set_ylabel(strong_features[1])
ax.set_zlabel("CO2 Emissions(g/km)")
ax.set_title(f'Relationship between {strong_features[0]}, {strong_features[1]}, and CO2 Emissions')
plt.show()

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (learning_rate / m) * X.T.dot(errors)
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

X_train_lr_bias = np.c_[np.ones(X_train_lr.shape[0]), X_train_lr]
X_test_lr_bias = np.c_[np.ones(X_test_lr.shape[0]), X_test_lr]
y_train_lr = y_train.values.reshape(-1, 1)
y_test_lr = y_test.values.reshape(-1, 1)

theta = np.zeros((X_train_lr_bias.shape[1], 1))
learning_rate = 0.003
iterations = 1000

theta_final, cost_history = gradient_descent(X_train_lr_bias, y_train_lr, theta, learning_rate, iterations)

plt.plot(range(iterations), cost_history)
plt.title("Cost Function vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

y_pred_lr_train = X_train_lr_bias.dot(theta_final)
y_pred_lr_test = X_test_lr_bias.dot(theta_final)
print("R^2 Score (Train):", r2_score(y_train_lr, y_pred_lr_train))
print("R^2 Score (Test):", r2_score(y_test_lr, y_pred_lr_test))

y_log = data["Emission Class"]
X_log = data[strong_features]
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=0.2)

for col in X_log_train.columns:
    mean = X_log_train[col].mean()
    range_ = X_log_train[col].max() - X_log_train[col].min()
    X_log_train[col] = (X_log_train[col] - mean) / range_
    X_log_test[col] = (X_log_test[col] - mean) / range_

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_log_loss(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    return -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def logistic_regression_sgd(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = sigmoid(X.dot(theta))
        errors = predictions - y
        theta -= (learning_rate / m) * X.T.dot(errors)
        cost_history.append(compute_log_loss(X, y, theta))
    return theta, cost_history

X_log_train_bias = np.c_[np.ones(X_log_train.shape[0]), X_log_train]
X_log_test_bias = np.c_[np.ones(X_log_test.shape[0]), X_log_test]
y_log_train = y_log_train.values.reshape(-1, 1)
y_log_test = y_log_test.values.reshape(-1, 1)

theta_log = np.zeros((X_log_train_bias.shape[1], 1))
learning_rate = 0.01
iterations = 1000

theta_log_final, log_loss_history = logistic_regression_sgd(X_log_train_bias, y_log_train, theta_log, learning_rate, iterations)

plt.plot(range(iterations), log_loss_history)
plt.title("Log Loss vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.show()

y_log_pred_test = sigmoid(X_log_test_bias.dot(theta_log_final)) >= 0.5
print("Accuracy (Test):", accuracy_score(y_log_test, y_log_pred_test))
