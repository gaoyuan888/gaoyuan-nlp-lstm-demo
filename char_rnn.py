import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

raw_text = open('input/Winston_Churchil.txt', encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
print(chars)
print('不重复的字符长度:{}'.format(len(chars)))
print('原文长度:{}'.format(len(raw_text)))
# 转换成字典
print(char_to_int)
print(int_to_char)

# 构造训练测试集
# 我们需要把我们的raw text变成可以用来训练的x,y:
# x 是前置字母们 y 是后一个字母
# 这里构建了每100个字符预测后一个字符
seq_length = 100
x = []
y = []

for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

print(x[:3])
print(y[:3])

n_patterns = len(x)
n_vocab = len(chars)
# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)

print(x[11])
print(y[11])

model = Sequential()
# 长短期神经网络（循环层）
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
# dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
# dropout是CNN中防止过拟合提高效果的一个大杀器
model.add(Dropout(0.5))
# 全连接层
model.add(Dense(y.shape[1], activation='softmax'))
# 优化器 我们使用’adam’优化器，损失函数为’categorical_crossentropy’。
model.compile(loss='categorical_crossentropy', optimizer='adam')
# 训练，
model.fit(x, y, nb_epoch=50, batch_size=4096)


# 我们来写个程序，看看我们训练出来的LSTM的效果：
def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    return res


def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c


# 好，写成一个大程序：
def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string


init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)
