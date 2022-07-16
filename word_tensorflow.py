import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import time

resultPath = 'results'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def getdata(rawtraindata, rawvalidata, rawtestdata):
    le = LabelEncoder()
    rawtraindata['region'] = le.fit_transform(rawtraindata['region'])
    rawvalidata['region'] = le.fit_transform(rawvalidata['region'])
    train_x, train_y = rawtraindata['tweet'].copy(), rawtraindata[
        'region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = rawtestdata['tweet'].copy()
    validation_x, validation_y = rawvalidata['tweet'].copy(), rawvalidata['region'].copy()
    return train_x, train_y, validation_x, validation_y, test_x,le


def saveResult(data, name):
    df = pd.DataFrame()
    df['region'] = data
    df['id'] = df.index + 1
    df.to_csv(resultPath + '/word_DNNresult' + name + '.csv', index=None)


def performanceMatrix(acts, pred):
    print("     the accuracy of validation test: " + str(accuracy_score(acts, pred)))
    print("     the f1 score of validation test: " + str(f1_score(acts, pred, average='macro')))
    print('')


def produce_result(msg, classifier, train_x, train_y, validation_x, validation_y, test_x):
    print("the performance matrix of " + msg + " : ")
    begin = time.time()
    print("the program start run at " + begin.__str__())
    classifier.fit(train_x, train_y)
    classifier_predict_validation = classifier.predict(validation_x)
    end = time.time()
    print("the program end run at " + end.__str__() + " , used: " + str(end - begin) + " seconds")
    performanceMatrix(validation_y, classifier_predict_validation)
    test_predict = classifier.predict(test_x)
    saveResult(test_predict, msg)

def make_predict(list, le):

    result= np.argmax(list, axis=1)
    result = le.inverse_transform(result)
    return result

def product_model_result(msg, model, test_x,le):
    print("the performance matrix of " + msg + " : ")
    test_predict = model.predict(test_x)
    print(test_predict)
    saveResult(make_predict(test_predict,le), msg)


# use the word of highest TFIDF value
def preprocesdata_tfidf(data):
    result = []
    for line in data:
        bestTFIDF = max(eval(line), key=lambda item: item[1])
        result.append(bestTFIDF[0])
    train_x = pd.DataFrame(result)
    return train_x


def preprocesdata_tfidf_pading(data, max_len):
    result = []
    for line in data:
        temp = eval(line)
        list = []
        for num in range(max_len):
            if num < len(temp):
                list.append(float(temp[num][0]))
                list.append(float(temp[num][1]))
            else:
                list.append(0.0)
                list.append(0.0)
        result.append(list)
    result = pd.DataFrame(result).astype('float64')
    return result


def readVocab():
    loadVocab = {}
    file = open('data/vocab.txt', 'r')
    for line in file.readlines():
        line = line.strip()
        key, value = line.split('\t')[0], line.split('\t')[1]
        loadVocab[key] = value
    file.close()
    return loadVocab



def create_model1(train_x, train_y, validation_x, validation_y, max_len):
    model_1 = Sequential([
        layers.Embedding(10000, 128, input_length=max_len * 2),
        layers.GlobalAveragePooling1D(),
        #layers.GlobalMaxPooling1D(),
        # tf.keras.layers.Dense(128, activation='relu'),
        # layers.Dropout(0.2),
        # tf.keras.layers.Dense(64, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    model_1.summary()
    model_1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_1.fit(train_x, train_y, epochs=6, validation_data=(validation_x, validation_y))
    return model_1


if __name__ == '__main__':
    data_train_count = pd.read_csv('data/train_count.csv')
    data_test_count = pd.read_csv('data/test_count.csv')
    data_vali_count = pd.read_csv('data/dev_count.csv')
    vocab = readVocab()
    train_x, train_y, validation_x, validation_y, test_x, le = getdata(data_train_count, data_vali_count, data_test_count)

    max_len = max(
        [len(eval(x)) for x in train_x] + [len(eval(x)) for x in validation_x] + [len(eval(x)) for x in test_x])

    train_x = preprocesdata_tfidf_pading(train_x, max_len)
    test_x = preprocesdata_tfidf_pading(test_x, max_len)
    validation_x = preprocesdata_tfidf_pading(validation_x, max_len)

    print(train_x)

    # produce_result(GaussianNB(), train_x, train_y, validation_x,validation_y)
    model1 = create_model1(train_x, train_y, validation_x, validation_y, max_len)
    product_model_result("model1", model1, test_x,le)