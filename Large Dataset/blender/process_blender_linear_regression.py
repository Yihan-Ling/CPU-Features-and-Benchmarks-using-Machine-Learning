import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

Process_count = []
seven_zip = []

input_path = './data/Blender Dataset.csv'

with open(input_path, mode='r') as file:
    csv_file = csv.DictReader(file)

    for lines in csv_file:
        Process = lines['Process']
        temp_list = []
        temp_list.append(int(Process))
        Process_count.append(temp_list)
        seven_zip.append(float(lines['Blender']))


c = np.array(Process_count)
s = np.array(seven_zip)

X_train = c
Y_train = s

linereg = linear_model.LinearRegression()
linereg.fit(X_train, Y_train)
y_pred = linereg.predict(X_train)
print("Linear Regression R2 score for training data is {}".format(linereg.score(X_train, Y_train)))
print("Linear Regression slope is {}".format(linereg.coef_))

plt.scatter(X_train,Y_train, color='black')
plt.plot(X_train, y_pred, color='blue', linewidth=3)
plt.title('Relationship between Process Size and Blender (Linear Regression model)')
plt.xlabel('Process (nm)')
plt.ylabel('Blender')
plt.xticks(())
plt.yticks(())
plt.show()