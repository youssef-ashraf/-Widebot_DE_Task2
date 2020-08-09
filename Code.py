import numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train = pd.read_csv("training.csv", ';')
# df_train = pd.DataFrame(train)  for examing the data
vaild = pd.read_csv("validation.csv", ';')
# df_vaild = pd.DataFrame(vaild)  for examing the data

train = train.dropna()
vaild = vaild.dropna()

train = train.values
vaild = vaild.values

x_train, y_train = train[:, 0:18], train[:, 18]
x_vaild, y_vaild = vaild[:, 0:18], vaild[:, 18]

encoder1 = LabelEncoder()
y_train = encoder1.fit_transform(y_train)
y_vaild = encoder1.fit_transform(y_vaild)

#I Had to OneHot Encode the String Variables:
encoder2 = OneHotEncoder(handle_unknown='ignore')

x = numpy.concatenate((x_train[:, (0, 3, 4, 5, 6, 8, 9, 11, 12, 16)], x_vaild[:, (0, 3, 4, 5, 6, 8, 9, 11, 12, 16)]))
this = encoder2.fit_transform(x)

x_train = numpy.concatenate((x_train[:, (1, 2, 7, 10, 13, 14, 15, 17)], this.toarray()[:1463, :]), axis=1)

x_vaild = numpy.concatenate((x_vaild[:, (1, 2, 7, 10, 13, 14, 15, 17)], this.toarray()[1463:, :]), axis=1)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print("Decision Tree: ", model.score(x_vaild, y_vaild))

model = SVC()
model.fit(x_train, y_train)

print("SVC: ", model.score(x_vaild, y_vaild))

# Decision Tree:  0.4880952380952381
# SVC:  0.4166666666666667
