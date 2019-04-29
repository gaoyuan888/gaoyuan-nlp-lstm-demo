import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

raw_text = ''
for file in os.listdir("input/"):
    if file.endswith(".txt"):
        raw_text += open("input/" + file, errors='ignore').read() + '\n\n'
# raw_text = open('../input/Winston_Churchil.txt').read()
raw_text = raw_text.lower()
# nltk 相关介绍 https://blog.csdn.net/zzulp/article/details/77150129
sentensor = nltk.data.load('english.pickle')
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

print(len(corpus))
print(corpus[:3])

w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)

print(w2v_model['office'])

raw_input = [item for sublist in corpus for item in sublist]
print(len(raw_input))

print(raw_input[12])

text_stream = []
vocab = w2v_model.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
len(text_stream)

seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])

print(x[10])
print(y[10])

print(len(x))
print(len(y))
print(len(x[12]))
print(len(x[12][0]))
print(len(y[12]))

model = Sequential()
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, nb_epoch=50, batch_size=4096)


def predict_next(input_array):
    x = np.reshape(input_array, (-1, seq_length, 128))
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream) - seq_length):]:
        res.append(w2v_model[word])
    return res


def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word


def generate_article(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' ' + n[0][0]
    return in_string


init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
article = generate_article(init)
print(article)
