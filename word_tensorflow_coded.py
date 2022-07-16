import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import time

import preprocess_tfidf
import preprocess_wordcount

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
    df.to_csv(resultPath + '/tfidf_coded_tensorflowresult' + name + '.csv', index=None)


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
        layers.Embedding(10000, 256, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(128, activation='relu'),
        # layers.Dropout(0.2),
        # tf.keras.layers.Dense(64, activation='relu'),
        # layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        # layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model_1.summary()
    model_1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_1.fit(train_x, train_y, epochs=5, validation_data=(validation_x, validation_y))
    return model_1

if __name__ == '__main__':
    train_x, train_y, validation_x, validation_y, test_x = preprocess_wordcount.readData()
    le = LabelEncoder()
    train_y= le.fit_transform(train_y)
    validation_y = le.transform(validation_y)
    max_len30 = 30
    max_len50 = 50
    max_len100 = 100
    max_len300 = 300
    selector = SelectFromModel(estimator=SGDClassifier()).fit(train_x, train_y)
    train_x = selector.transform(train_x)
    validation_x = selector.transform(validation_x)
    test_x = selector.transform(test_x)
    print(train_x)
    print(train_x.shape)
    max_len = train_x.shape[1]
    # produce_result(GaussianNB(), train_x, train_y, validation_x,validation_y)
    model1 = create_model1(train_x, train_y, validation_x, validation_y, max_len)
    product_model_result("model1", model1, test_x,le)