import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

L2_count = []
blender = []

input_path = './data/Blender Dataset.csv'

with open(input_path, mode='r') as file:
    csv_file = csv.DictReader(file)
    for lines in csv_file:
        cache = lines['L2 Cache']
        temp_list = []
        temp_list.append(float(cache))
        L2_count.append(temp_list)
        blender.append(float(lines['Blender']))


c = np.array(L2_count)
s = np.array(blender)

# x,y = make_regression(n_samples=20, n_features=1, noise=10)

X_train = c
Y_train = s

linereg = linear_model.LinearRegression()
linereg.fit(X_train, Y_train)
y_pred = linereg.predict(X_train)
print("Linear Regression R2 score for training data is {}".format(linereg.score(X_train, Y_train)))
print("Linear Regression slope is {}".format(linereg.coef_))


plt.scatter(X_train,Y_train, color='black')
plt.plot(X_train, y_pred, color='blue', linewidth=3)
plt.title('Relationship between L2 Cache and blender (Linear Regression model)')
plt.xlabel('L2 Cache (MB)')
plt.ylabel('blender')
plt.xticks(())
plt.yticks(())
plt.show()