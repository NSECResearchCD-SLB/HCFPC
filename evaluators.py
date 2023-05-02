from sklearn.model_selection import KFold
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def ten_fold_svm(x,y):
    x=np.array(x)
    y=np.array(y)
    cls=svm.SVC()
    kf=KFold(n_splits=10)
    total=0
    match=0
    for train_index, test_index in kf.split(x):
        cls.fit(x[train_index],y[train_index])
        for i in test_index :
            prediction=cls.predict(x[i].reshape(1,-1))
            if prediction == int(y[i]):
                match=match+1
            total=total+1
    accuracy=match/total
    return accuracy

def ten_fold_dt(x,y):
    x=np.array(x)
    y=np.array(y)
    cls=tree.DecisionTreeClassifier()
    kf=KFold(n_splits=10)
    total=0
    match=0
    for train_index, test_index in kf.split(x):
        cls.fit(x[train_index],y[train_index])
        for i in test_index :
            prediction=cls.predict(x[i].reshape(1,-1))
            if prediction == int(y[i]):
                match=match+1
            total=total+1
    accuracy=match/total
    return accuracy

def ten_fold_sgdc(x,y):
    x=np.array(x)
    y=np.array(y)
    cls=SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    kf=KFold(n_splits=10)
    total=0
    match=0
    for train_index, test_index in kf.split(x):
        cls.fit(x[train_index],y[train_index])
        for i in test_index :
            prediction=cls.predict(x[i].reshape(1,-1))
            if prediction == int(y[i]):
                match=match+1
            total=total+1
    accuracy=match/total
    return accuracy

def ten_fold_knn(x,y):
    x=np.array(x)
    y=np.array(y)
    cls=KNeighborsClassifier(n_neighbors=3)
    kf=KFold(n_splits=10)
    total=0
    match=0
    for train_index, test_index in kf.split(x):
        cls.fit(x[train_index],y[train_index])
        for i in test_index :
            prediction=cls.predict(x[i].reshape(1,-1))
            if prediction == int(y[i]):
                match=match+1
            total=total+1
    accuracy=match/total
    return accuracy
