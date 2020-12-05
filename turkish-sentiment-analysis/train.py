from generate_data import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

X,y=(generateData())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=42,shuffle=True)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


model = Pipeline([
        ('vectorizer', CountVectorizer()),
        # ('classifier', SVC())
        ('classifier', LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000000000))
        
    ])

model.fit(X_train,y_train)


print('train score=',model.score(X_train, y_train))
print('test score=',model.score(X_test, y_test))



y_hat=model.predict(X_test)
confusion = confusion_matrix(y_test, y_hat)
print(confusion)


print(classification_report(y_test, y_hat, target_names=['0', '1']))



