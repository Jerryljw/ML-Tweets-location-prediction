import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import time

resultPath = 'results'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def getdata(rawtraindata, rawvalidata, rawtestdata):
    train_x, train_y = rawtraindata['tweet'].copy(), rawtraindata[
        'region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = rawtestdata['tweet'].copy()
    validation_x, validation_y = rawvalidata['tweet'].copy(), rawvalidata['region'].copy()
    return train_x, train_y, validation_x, validation_y, test_x


def saveResult(data, name):
    df = pd.DataFrame()
    df['region'] = data
    df['id'] = df.index + 1
    df.to_csv(resultPath + '/tfidf_Baseline_result' + name + '.csv', index=None)


# use the word of highest TFIDF value
def preprocesdata_tfidf_1left(data):
    result = []
    for line in data:
        bestTFIDF = max(eval(line), key=lambda item: item[1])
        result.append(bestTFIDF[0])
    train_x = pd.DataFrame(result)
    return train_x


def preprocesdata_tfidf_maxCreates(data, max_len):
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


if __name__ == '__main__':
    data_train_tfidf = pd.read_csv('data/train_tfidf.csv')
    data_test_tfidf = pd.read_csv('data/test_tfidf.csv')
    data_vali_tfidf = pd.read_csv('data/dev_tfidf.csv')
    vocab = readVocab()
    train_x, train_y, validation_x, validation_y, test_x = getdata(data_train_tfidf, data_vali_tfidf, data_test_tfidf)
    max_len = max(
        [len(eval(x)) for x in train_x] + [len(eval(x)) for x in validation_x] + [len(eval(x)) for x in test_x])
    train_x = preprocesdata_tfidf_1left(train_x)
    test_x = preprocesdata_tfidf_1left(test_x)
    validation_x = preprocesdata_tfidf_1left(validation_x)
    print(train_x)
    produce_result("naive_bayes", BernoulliNB(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("neural_network", MLPClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("k_nearest_neighbour", KNeighborsClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("Stochastic_Gradient_Descent", SGDClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("AdaBoost", AdaBoostClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("Random_Forest", RandomForestClassifier(), train_x, train_y, validation_x, validation_y, test_x)
