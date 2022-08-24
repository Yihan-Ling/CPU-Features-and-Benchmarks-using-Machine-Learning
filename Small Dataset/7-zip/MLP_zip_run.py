from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd

input_path = './data/7-zip Dataset.csv'
df_raw = pd.read_csv(input_path, sep=',')

l2 = df_raw['L2 Cache (MB)'].values.tolist()
l3 = df_raw['L3 Cache (MB)'].values.tolist()
tpd = df_raw['TDP (Watt)'].values.tolist()
mhz = df_raw['MHz'].values.tolist()
turbo = df_raw['Turbo'].values.tolist()
cores = df_raw['Cores'].values.tolist()
threads = df_raw['Threads'].values.tolist()
process = df_raw['Process (nm)'].values.tolist()
x = np.array([l2, l3, tpd, mhz, turbo, cores, threads, process]).transpose()
print(x)
y = df_raw['7-Zip'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 16, 4),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_mlp.fit(X_train, y_train)

pred_train = model_mlp.predict(X_train)
mse_train = mean_squared_error(pred_train, y_train)
print("Train MSE: ", mse_train)

pred_test = model_mlp.predict(X_test)
mse_test = mean_squared_error(pred_test, y_test)
print("Test MSE: ", mse_test)

mlp_score = model_mlp.score(X_test, y_test)
print('R-Squared Score', mlp_score)

xx = range(0, len(y_train))
plt.figure(figsize=(8, 6))
plt.scatter(xx, y_train, marker="o", color="green", label="Ground Truth", linewidths=3)
plt.plot(xx, pred_train, marker="^", color="orange", label="Prediction", linewidth=1)
plt.title('Fitting on Train')
plt.xlabel('Order in Training Dataset')
plt.ylabel('7-zip')
plt.legend()
plt.show()

xx = range(0, len(y_test))
plt.figure(figsize=(8, 6))
plt.scatter(xx, y_test, marker="o", color="green", label="Ground Truth", linewidths=3)
plt.plot(xx, pred_test, marker="^", color="orange", label="Prediction", linewidth=1)
plt.title('Prediction on Test')
plt.xlabel('Order in Test Dataset')
plt.ylabel('7-zip')
plt.legend()
plt.show()