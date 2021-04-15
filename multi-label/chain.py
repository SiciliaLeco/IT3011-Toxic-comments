from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

train=pd.read_csv("/Users/liqilin/PycharmProjects/untitled/toxic_comment_classifier/data/train.csv")

y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X = train.comment_text.apply(lambda x: np.str_(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = TfidfVectorizer(max_features=30000)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
classifier = ClassifierChain(GaussianNB())
classifier.fit(X_train_dtm, y_train)

predictions = classifier.predict(X_test_dtm)

print("accracy = ", metrics.accuracy_score(predictions, y_test))
print("f1 score= ", metrics.f1_score(predictions, y_test, average="micro"))