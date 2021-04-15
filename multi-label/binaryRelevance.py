from skmultilearn.problem_transform import BinaryRelevance
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

train=pd.read_csv("/Users/liqilin/PycharmProjects/untitled/toxic_comment_classifier/data/train.csv")
att=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X = train.comment_text.apply(lambda x: np.str_(x))

# train and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = TfidfVectorizer(max_features=30000)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
classifier = BinaryRelevance(MultinomialNB())

# train
classifier.fit(X_train_dtm, y_train)
# predict
predictions = classifier.predict(X_test_dtm)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
# print('Test accuracy is {}'.format(f1_score(y_test, predictions)))
Y_pred_1 = [[] for i in range(6)]
Y_test_1 = [[] for j in range(6)]

for item in range(len(predictions)):
    for i in range(len(predictions[item])):
        if predictions[item][i] > 0.2:
            Y_pred_1[i].append(1)
        else:
            Y_pred_1[i].append(1)
#
print(Y_pred_1)
print(len(Y_pred_1))
print(len(Y_pred_1[0]))

for index in y_test.index:
    for i in range(6):
        Y_test_1[i].append(train.loc[index, att[i]])

print(Y_test_1)
print(len(Y_test_1))
print(len(Y_test_1[0]))

for i in range(6):
    print(f1_score(Y_pred_1[i], Y_test_1[i]))