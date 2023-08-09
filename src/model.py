from sklearn import tree
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import metrics
from os import path

def fit():
    # clean data
    a = pd.read_csv("term-deposit-marketing-2020.csv")
    a['job'].replace(['management', 'technician', 'entrepreneur', 'blue-collar',
       'unknown', 'retired', 'admin', 'services', 'self-employed',
       'unemployed', 'housemaid', 'student'],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
    a['marital'].replace(['married', 'single', 'divorced'],
                        [0, 1, 2], inplace=True)
    a['education'].replace(['tertiary', 'secondary', 'unknown', 'primary'],
                        [0, 1, 2, 3], inplace=True)
    a['default'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    a['housing'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    a['loan'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    a['contact'].replace(['unknown', 'cellular', 'telephone'],
                        [0, 1, 2], inplace=True)
    a['month'].replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
       'mar', 'apr'],
                        [5, 6, 7, 8, 10, 11, 12, 1, 2, 3, 4], inplace=True)
    a['y'].replace(["no", "yes"],
                        [0,1], inplace=True)
    X = a.drop(["y"], axis=1)
    y = a["y"]

    kf = KFold(n_splits=5)
    d = {}
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        d["train{0}".format(i)] = train_index
        d["test{0}".format(i)] = test_index

    
    f1 = []
    accuracy = []
    conf_matrix = []
    for i in range(5):
        train = X.loc[d["train{0}".format(i)]]
        test = X.loc[d["test{0}".format(i)]]
        y_train = y[d["train{0}".format(i)]]
        y_test = y[d["test{0}".format(i)]]
        y_train = y_train.reset_index(drop=True)
        ind = []
        for j in range(32000):
            if y_train[j] == 1:
                ind.append(j)
        new_X = train.iloc[ind]
        ind1 = []
        for k in range(32000):
            if y_train[k] == 0:
                ind1.append(k)
        new_X1 = train.iloc[ind1]
        neg_sample = 8000-len(ind)
        new_X2 = new_X1.sample(n=neg_sample, replace=False, random_state=0)
        new_X3 = pd.concat([new_X,new_X2])
        y_train = []
        for i in range(len(ind)):
            y_train.append(1)
        for i in range(neg_sample):
            y_train.append(0)
        clf = RandomForestClassifier(random_state = 0)
        clf.fit(new_X3, y_train)
        y_pred = clf.predict(test)
        f1.append(f1_score(y_test, y_pred))
        accuracy.append(clf.score(test, y_test))
        conf_matrix.append(confusion_matrix(y_test, y_pred))

    return f1, accuracy, conf_matrix