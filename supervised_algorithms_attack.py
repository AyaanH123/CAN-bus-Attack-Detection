import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import timeit as t

names = ['a','b','c','d','e','f','g','h','i','j','k','l']
dataset_dos = pd.read_csv('DoS_dataset.csv', header=None, names=names)
dataset_fuzzy = pd.read_csv('Fuzzy_dataset.csv', header=None, names=names)
dataset_gear = pd.read_csv('gear_dataset.csv', header=None, names=names)
dataset_rpm = pd.read_csv('RPM_dataset.csv', header=None, names=names)

dataset = pd.concat([dataset_dos, dataset_gear, dataset_rpm])
dataset = dataset.reset_index(drop=True)

dataset['b'] = dataset['b'].apply(int, base=16)
dataset['d'] = dataset['d'].apply(int, base=16)
dataset['e'] = dataset['e'].apply(int, base=16)
dataset['f'] = dataset['f'].apply(int, base=16)
dataset['g'] = dataset['g'].apply(int, base=16)
dataset['h'] = dataset['h'].apply(int, base=16)
dataset['i'] = dataset['i'].apply(int, base=16)
dataset['j'] = dataset['j'].apply(int, base=16)
dataset['k'] = dataset['k'].apply(int, base=16)
        
array = dataset.values

X = array[:,0:11]
Y = array[:,11]

validation_size = 0.4
seed = 2

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'
# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []

#Testing multiple models consequently

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

start = t.default_timer()
alg = LinearDiscriminantAnalysis()
alg.fit(X_train, Y_train)
predictions = alg.predict(X_validation)
stop = t.default_timer()
time_elapsed = stop - start
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(time_elapsed)