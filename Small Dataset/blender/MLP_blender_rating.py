from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
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

x_array = [x[0:17], x[17:34], x[34:51], x[51:68], x[68:85], x[85:102], x[102:119], x[119:136], x[136:153], x[153:]]
y_array = [y[0:17], y[17:34], y[34:51], y[51:68], y[68:85], y[85:102], y[102:119], y[119:136], y[136:153], y[153:]]


model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 16, 4),  activation='relu', solver='adam', alpha=0.0001, batch_size=20,
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True,
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
    print('R-Squared', reg_score)


