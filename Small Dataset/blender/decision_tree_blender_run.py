import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

input_path = './data/Blender Dataset.csv'
df_raw = pd.read_csv(input_path, sep=',')

l2 = df_raw['L2 Cache'].values.tolist()
l3 = df_raw['L3 Cache'].values.tolist()
tpd = df_raw['TDP'].values.tolist()
mhz = df_raw['MHz'].values.tolist()
turbo = df_raw['Turbo'].values.tolist()
cores = df_raw['Cores'].values.tolist()
threads = df_raw['Threads'].values.tolist()
process = df_raw['Process'].values.tolist()
x = np.array([l2, l3, tpd, mhz, turbo, cores, threads, process]).transpose()
print(x)
y = df_raw['Blender'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X_train, y_train)
pred_train = regressor.predict(X_train)
mse_train = mean_squared_error(pred_train, y_train)
print("Train MSE: ", mse_train)

pred_test = regressor.predict(X_test)
mse_test = mean_squared_error(pred_test, y_test)
print("Test MSE: ", mse_test)

reg_score = regressor.score(X_test, y_test)
print('sklearn多层感知器-回归模型得分', reg_score)#预测正确/总数

xx = range(0, len(y_train))
plt.figure(figsize=(8, 6))
plt.scatter(xx, y_train, marker="o", color="green", label="Ground Truth", linewidths=3)
plt.plot(xx, pred_train, marker="^", color="orange", label="Prediction", linewidth=1)
plt.title('Fitting on Train')
plt.xlabel('Order in Training Dataset')
plt.ylabel('Blender')
plt.legend()
plt.show()

xx = range(0, len(y_test))
plt.figure(figsize=(8, 6))
plt.scatter(xx, y_test, marker="o", color="green", label="Ground Truth", linewidths=3)
plt.plot(xx, pred_test, marker="^", color="orange", label="Prediction", linewidth=1)
plt.title('Prediction on Test')
plt.xlabel('Order in Test Dataset')
plt.ylabel('Blender')
plt.legend()
plt.show()


