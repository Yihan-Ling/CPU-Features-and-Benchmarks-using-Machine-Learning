from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

input_path = './data/7-zip Dataset.csv'
df_raw = pd.read_csv(input_path, sep=',')

df_raw = shuffle(df_raw)

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


x_array = [x[0:24], x[24:48], x[48:72], x[72:96], x[96:120], x[120:144], x[144:168], x[168:192], x[192:216], x[216:]]
y_array = [y[0:24], y[24:48], y[48:72], y[72:96], y[96:120], y[120:144], y[144:168], y[168:192], y[192:216], y[216:]]


model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 16, 4),  activation='relu', solver='adam', alpha=0.0001, batch_size=16,
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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

    model_mlp.fit(X_train, y_train)
    pred_train = model_mlp.predict(X_train)
    mse_train = mean_squared_error(pred_train, y_train)
    print("Train MSE: ", mse_train)

    pred_test = model_mlp.predict(X_test)
    mse_test = mean_squared_error(pred_test, y_test)
    print("Test MSE: ", mse_test)

    reg_score = model_mlp.score(X_test, y_test)
    print('R-Squared Score', reg_score)
