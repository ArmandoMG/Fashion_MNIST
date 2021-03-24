# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:07:51 2021

@author: arman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data  = pd.read_csv('fashion-mnist_test.csv')

#Checking for nulls, always a good practice
train_data.isnull().sum()
test_data.isnull().sum()

#Since we have a high amount of data we cant perform any fancy model selection technic or hyperparameter tuning, since it'd take a high a mount of time
#and your time as a programmer is very valuable :). So i'll start with a reasonable amount of data in order to get an idea of which model to pick.

#We'll try 3 models
#Random Forest Classifier
#K Nearest neighbors
#Support Vector Machine

#We wont use Logistic regression nor naive bayes, since we dont want to know the exact probabity. Also, Even though logistic regression works fine for multiple labels predictions, it works the best when is just binary
#NOTE: I tried a simple LR model in another file, and it is a good model, but as we'll see, it gets outperformed by other models.

#Splitting the data (we start with 10000 just to give us an idea of which model to pick)
X_train    = train_data.iloc[:10000, 1:].values
y_train    = train_data.iloc[:10000, 0].values
X_test     = test_data.iloc[:, 1:].values
y_test     = test_data.iloc[:, 0].values

#===================================================
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#===================================================
#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#===================================================
#SVM
from sklearn.svm import SVC
classifier = SVC(gamma="scale", kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#===================================================

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
accuracy_score(y_test, y_pred)

#Since SVM was the best model for this case, I'll pick that model for further analysis and tuning
#Lets put some more data in the train dataset
X_train    = train_data.iloc[:15000, 1:].values
y_train    = train_data.iloc[:15000, 0].values

from sklearn.model_selection import GridSearchCV



params = {
 		  'C': [0.01,0.1,1,10,50,100],
 		  'kernel': ['rbf','sigmoid'],
 		  'gamma':['scale']
 		 }
classifier = SVC(**params, probability=True)

classifier_opt = GridSearchCV(classifier, param_grid=params, scoring='neg_log_loss', n_jobs=-1, cv=2, verbose=10)
classifier_opt.fit(X_train, y_train)

classifier_opt.best_params_
classifier_opt.best_score_

from sklearn.svm import SVC
classifier = SVC(gamma="scale", kernel = 'rbf', C=10, random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)

#88.35% its a pretty good model for predictions, but maybe we can improve a little bit more in both efficiency and efectivity
from sklearn.model_selection import cross_validate
cv_results = cross_validate(classifier, X_train, y_train, cv=3)
print(f"Mean accuracy:  {cv_results['test_score'].mean()}")

from sklearn.decomposition import PCA
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(n_components=400)
pca.fit(X_train)
pca_X_train = pca.transform(X_train)
pca_X_test = pca.transform(X_test)

classifier = SVC(gamma='scale',kernel='rbf',C=10)

classifier.fit(X_train,y_train)

preds = classifier.predict(X_test)

print(f"Test acc {accuracy_score(preds,y_test)}")

#FAILURES OR BAD PRACTICES TO LEARN:
    #We could do graph that tells us the accuracy in function of the amount of data given to the model, in order to know how much data we can give to the model in order to be as fast as possible
    #It would be nice to set a constant amount of data. The same amount of data when we are choosing our model as when we are tuning the best model
    #We can do a better work narrowing down the optimal value for C