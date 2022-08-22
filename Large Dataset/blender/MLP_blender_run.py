from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd

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
# # alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# # hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
# #clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# #'identity'，无操作**，对实现线性瓶颈很有用，返回f（x）= x
# #'logistic'，logistic sigmoid函数，返回f（x）= 1 /（1 + exp（-x））。
# #'tanh'，双曲tan函数，返回f（x）= tanh（x）。
# #'relu'，整流后的线性单位函数，返回f（x）= max（0，x）
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
print('sklearn多层感知器-回归模型得分', mlp_score)#预测正确/总数

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