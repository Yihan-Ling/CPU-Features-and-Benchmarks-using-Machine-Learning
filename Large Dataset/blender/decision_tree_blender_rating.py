import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

input_path = './data/Blender Dataset.csv'
df_raw = pd.read_csv(input_path, sep=',')

df_raw = shuffle(df_raw)

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


x_array = [x[0:24], x[24:48], x[48:72], x[72:96], x[96:120], x[120:144], x[144:168], x[168:192], x[192:216], x[216:]]
y_array = [y[0:24], y[24:48], y[48:72], y[72:96], y[96:120], y[120:144], y[144:168], y[168:192], y[192:216], y[216:]]


regressor = DecisionTreeRegressor(random_state=0)

for i in range(10):
    print("Trial ", i+1)
    X_test = np.array(x_array[i])
    y_test = np.array(y_array[i])
    X_train = np.array([x[0]])
    y_train = np.array([])
    for j in range(10):
        if j == i:
            continue
        X_train = np.append(X_train,x_array[j],axis=0)
        y_train = np.append(y_train,y_array[j],axis=0)

    X_train = np.delete(X_train,0, axis=0)

    regressor.fit(X_train, y_train)
    print(regressor.feature_importances_)
    pred_train = regressor.predict(X_train)
    mse_train = mean_squared_error(pred_train, y_train)
    print("Train MSE: ", mse_train)

    pred_test = regressor.predict(X_test)
    mse_test = mean_squared_error(pred_test, y_test)
    print("Test MSE: ", mse_test)

    reg_score = regressor.score(X_test, y_test)
    print('R-Squared Score', reg_score)#预测正确/总数

