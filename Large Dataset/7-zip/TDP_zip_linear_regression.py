import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

TDP_count = []
seven_zip = []

input_path = './data/7-zip Dataset.csv'

with open(input_path, mode='r') as file:
    csv_file = csv.DictReader(file)

    for lines in csv_file:
        TDP = lines['TDP (Watt)']
        temp_list = []
        temp_list.append(float(TDP))
        TDP_count.append(temp_list)
        seven_zip.append(float(lines['7-Zip']))


c = np.array(TDP_count)
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
plt.title('Relationship between TDP and 7-zip (Linear Regression model)')
plt.xlabel('TDP (Watts)')
plt.ylabel('7-zip')
plt.xticks(())
plt.yticks(())
plt.show()