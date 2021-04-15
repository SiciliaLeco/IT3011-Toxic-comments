from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

train=pd.read_csv("/Users/liqilin/PycharmProjects/untitled/toxic_comment_classifier/data/train.csv")
att=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X = train.comment_text.apply(lambda x: np.str_(x))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = TfidfVectorizer(max_features=30000)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

classifier_new = MLkNN(10)
# Note that this classifier can throw up errors when handling sparse matrices.
x_train = lil_matrix(X_train_dtm).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(X_test_dtm).toarray()
# train
classifier_new.fit(X_train_dtm, y_train)
# predict
predictions_new = classifier_new.predict(X_test_dtm)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions_new))
print("f1_score = ", f1_score(y_test,predictions_new,average='macro'))
print("\n")
