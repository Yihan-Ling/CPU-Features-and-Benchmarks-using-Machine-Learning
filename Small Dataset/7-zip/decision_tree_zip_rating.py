import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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


x_array = [x[0:18], x[18:36], x[36:54], x[54:72], x[72:90], x[90:108], x[108:126], x[126:144], x[144:162], x[162:]]
y_array = [y[0:18], y[18:36], y[36:54], y[54:72], y[72:90], y[90:108], y[108:126], y[126:144], y[144:162], y[162:]]


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
    print('R-Squared Score', reg_score)

