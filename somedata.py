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
vocab_to_int = dict((v, k) for k,v in enumerate(vocab))
n_chars = len(raw_text)
n_vocab = len(vocab_to_int)

seq_len = 10
dataX = np.array([])
dataY = np.array([])

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i + seq_len]

    seq_out = raw_text[i+seq_len]
    # print seq_in, seq_out
    dataX = np.append(dataX, [vocab_to_int[char] for char in seq_in])
    dataY = np.append(dataY, [vocab_to_int[seq_out]])

dataX = np.reshape(dataX, (len(dataX) / seq_len, seq_len))

dataX = dataX / n_vocab

dataY = np_utils.to_categorical(dataY)
print dataY.shape

model = Sequential()
model.add(Dense(32, input_shape=(dataX.shape[1], )))
model.add(Activation('sigmoid'))

model.add(Dense(dataY.shape[1]))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(dataX, dataY, epochs=10, batch_size=10, callbacks=callbacks_list)
