'''
simpler version of
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
'''

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils

raw_text = open('abcd.txt').read().lower()
vocab = sorted(set(raw_text.lower()))
vocab_to_int = dict((v, k) for k, v in enumerate(vocab))
int_to_vocab = dict((k, v) for k, v in enumerate(vocab))

n_chars = len(raw_text)
n_vocab = len(vocab_to_int)

seq_len = 10
dataX = np.array([])
dataY = np.array([])

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i + seq_len]
    seq_out = raw_text[i + seq_len]
    dataX = np.append(dataX, [vocab_to_int[char] for char in seq_in])
    dataY = np.append(dataY, [vocab_to_int[seq_out]])

dataX = np.reshape(dataX, (len(dataX) / seq_len, seq_len))

origx = dataX
dataX = dataX / n_vocab
dataY = np_utils.to_categorical(dataY)

model = Sequential()
model.add(Dense(32, input_shape=(dataX.shape[1],)))
model.add(Activation('sigmoid'))

model.add(Dense(dataY.shape[1]))
model.add(Activation('softmax'))

filename = 'weights-improvement-09-0.9826.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(dataX) - 1)
tp = dataX[start].reshape(1, seq_len)
seed = ''.join([int_to_vocab[value] for value in origx[start]])

for i in range(50):
    prediction = model.predict(tp)
    int_result = np.argmax(prediction)
    result = int_to_vocab[int_result]
    print result,
    seed = seed[1:len(seed)] + result
    int_rep = np.array([float(vocab_to_int[x]) for x in seed])
    tp = int_rep.reshape(1, seq_len)
    tp /= len(vocab)
