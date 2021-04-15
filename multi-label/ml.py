import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

# define functions for different vectorizors and classifiers
def test_vec_classifer(vect, clf):
    print('Features: ', X_train_dtm.shape[1])
    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

train=pd.read_csv("/Users/liqilin/PycharmProjects/untitled/toxic_comment_classifier/data/train.csv")

y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X = train.comment_text.apply(lambda x: np.str_(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = TfidfVectorizer(max_features=30000)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


for category in categories:
    print('**Processing {} comments...**'.format(category))

    LogReg_pipeline.fit(X_train_dtm, y_train[category])
    prediction = LogReg_pipeline.predict(X_test_dtm)
    print('Test accuracy is {}'.format(metrics.f1_score(y_test[category], prediction)))
    print("\n")