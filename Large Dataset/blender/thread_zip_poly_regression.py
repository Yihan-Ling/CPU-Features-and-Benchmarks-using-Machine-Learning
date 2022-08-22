import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import csv



thread_count = []
seven_zip = []

input_path = './data/7-zip Dataset.csv'

with open(input_path, mode='r') as file:
    csv_file = csv.DictReader(file)

    for lines in csv_file:
        core_thread = lines['Cores\xa0Threads']
        core_thread = core_thread.split(' ')
        if len(core_thread) >= 2:
            temp_list = []
            temp_list.append(int(core_thread[1]))
            thread_count.append(temp_list)
            # temp_list = []
            # temp_list.append(lines['7-Zip'])
            seven_zip.append(float(lines['7-Zip']))


t = np.array(thread_count)
s = np.array(seven_zip)

# x,y = make_regression(n_samples=20, n_features=1, noise=10)

X_train = t
Y_train = s
X_test = t[-20:]
Y_test = s[-20:]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X_train)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, Y_train)
y_predicted = poly_reg_model.predict(poly_features)

plt.scatter(X_train,Y_train)
plt.plot(X_train,y_predicted)
plt.show()

