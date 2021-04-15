import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout, Activation, Input
from keras.layers import Bidirectional, GlobalMaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
######### data retrivial  ##########

df = pd.read_csv("train.csv")
X = df.comment_text
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
tokenizer = Tokenizer(num_words = 40000)
train_tokenized = tokenizer.texts_to_sequences(X)
train_padded = pad_sequences(train_tokenized, maxlen=150)
print("training data, shape:", train_padded.shape)

######### build LSTM model #########

class lstm_model(object):
    def __init__(self,max_features=20000):
        self.max_features=max_features
        self.model=Sequential()
        self.model.add(Input(shape=200,))
        self.model.add(Embedding(self.max_features,128))
        self.model.add(Bidirectional(LSTM(50, return_sequences=True,recurrent_dropout=0.2)))
        self.model.add(Bidirectional(LSTM(50, return_sequences=True,recurrent_dropout=0.2)))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(6,activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    def fit(self,X,y,batchsize,epochs):
        X_train, X_val, Y_train, Y_val = train_test_split(X,
                                                          y, shuffle=True, test_size=0.25, random_state=1)
        self.model.fit(X_train, Y_train,batchsize,epochs,validation_data=(X_val, Y_val))

    def predict(self,data):
        return self.model.predict(data)

    def predict_classes(self,data):
        return self.model.predict_classes(data)

###### generate train-test dataset ######

model = lstm_model()
X_train, X_test, Y_train, Y_test = train_test_split(train_padded,
                                                    y, shuffle=True, test_size=0.2, random_state=1)

###### train test progress ######
THRESHOLD = 0.5
BATCH_SIZE = 256
EPOCH = 2

model.fit(X_train, Y_train, epochs=1, batchsize=100)

Y_pred = model.predict(X_test)
Y_pred = list(Y_pred)
Y_pred_1 = []
for item in range(len(Y_pred)):
    cur_list = []
    for i in Y_pred[item]:
        if i > THRESHOLD:
            cur_list.append(1)
        else:
            cur_list.append(0)
    Y_pred_1.append(cur_list)

print("Accuracy = ", accuracy_score(Y_test, Y_pred_1))
print("F1Score = ", f1_score(Y_test, Y_pred_1))