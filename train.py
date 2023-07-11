import numpy as np
from numpy.core.records import ndarray
from sklearn import datasets

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNNimp import KNNimp
import pandas as pd
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
data = pd.read_csv("cardata.csv")
print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""""
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""""

model = KNNimp(k=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
###true positive
acc=0
###specificnost
spec=0
###senzitivnost
sens=0

predicted = model.predict(X_test)
for x in range(len(X_test)):
    if(predicted[x]==y_test[x]):
        acc=acc+1
    if (predicted[x] > y_test[x]):
        spec = spec + 1
    if (predicted[x] < y_test[x]):
        sens = sens + 1


vm=acc
acc=acc/len(X_test)
##sens=sens/2
###names = ["ko babičin berlingo", "ko moj fićo", "spravi me do maribora pa nazaj", "ko sosedov mercedes"]
names = ["nesprejemljivo", "sprejemljivo", "dobro", "odlično"]
for x in range(len(X_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])
    ##n = model.kneighbors([X_test[x]], 7, True)
    ##print("N: ", n)


##acc = np.sum(predicted == y_test) / len(y_test)

Accuracy = metrics.accuracy_score(y_test, predicted)
print("Točnost: ",Accuracy)
##vm=truepositive to kar je res
##sens=false negative (false negative-underestimation: ", sens, "
##spec=false positive (false positive-overestimation: ",spec, "
Sensitivity_recall = vm/(vm+sens)
print("Senzitivnost:     ",round(Sensitivity_recall,5) )
##true negative
tn=len(X_test)-sens-spec
Specificity = tn/(tn+spec)
print("Specificnost:     ", round(Specificity,5) )


print("Priklic: ",round(Sensitivity_recall,5))

Precision = vm/(vm+spec)
print("Preciznost: ",round(Precision,5))

###Matrika zmede




print(classification_report(predicted, y_test))


conf_matrix = confusion_matrix(y_true=predicted, y_pred=y_test)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predvidevano stanje', fontsize=18)
plt.ylabel('Dejansko stanje', fontsize=18)
plt.title('Matrika Zmede', fontsize=18)
plt.show()




