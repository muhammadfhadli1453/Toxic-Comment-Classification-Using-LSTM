# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:41:41 2019

@author: Muhammad Fhadli
"""

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten

train = pd.read_csv(r'C:\Users\Muhammad Fhadli\Documents\Spyder\Jigsaw\Data\train.csv')
test = pd.read_csv(r'C:\Users\Muhammad Fhadli\Documents\Spyder\Jigsaw\Data\test.csv')

label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COMMENT = 'comment_text'

def get_list(data):
    data = data.fillna("unknown")
    return data.tolist()

def tokenize_reshape(data):
    data = tok.texts_to_matrix(data)
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))

x_train = get_list(train[COMMENT])
x_test = get_list(test[COMMENT])

tok = Tokenizer(num_words=1000)
tok.fit_on_texts(x_train)

x_train = tokenize_reshape(x_train)
x_test = tokenize_reshape(x_test)

model = Sequential()
model.add(LSTM(200, input_shape=(1, x_train.shape[2]), return_sequences=True))
model.add(LSTM(200, return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

result = np.zeros((len(x_test), len(label)))

for i, j in enumerate(label):
    print("Train ",j)
    y_train = train[j].values
    model.fit(x_train, y_train, epochs=10)
    result[:,i] = model.predict_proba(x_test)[:,0]
    
submission = pd.DataFrame(test['id'])
result = pd.DataFrame(result, columns=label)    
submission = pd.concat([submission, result], axis=1)
submission.to_csv(r'C:\Users\Muhammad Fhadli\Documents\Spyder\Jigsaw\Data\submission1.csv', index=False)