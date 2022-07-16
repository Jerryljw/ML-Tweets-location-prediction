import numpy as np
import pandas as pd





def readData():
    print("loading data......")
    data_train_wc = pd.read_csv('data/train_count.csv')
    data_test_wc = pd.read_csv('data/test_count.csv')
    data_vali_wc = pd.read_csv('data/dev_count.csv')
    train_x, train_y, validation_x, validation_y, test_x = getdata(data_train_wc,data_vali_wc,data_test_wc)
    vocabs = readVocab()
    train_x = coding_features(train_x,vocabs)
    test_x = coding_features(test_x, vocabs)
    validation_x = coding_features(validation_x, vocabs)
    print("training data: ")
    print(train_x)
    print("training labels: ")
    print(train_y)
    return train_x, train_y, validation_x, validation_y, test_x


def coding_features(features,vocabs):
    temp = []
    for i in range(len(vocabs)):
        temp.append(0)

    result = []
    for i in range(len(features)):
        temp1 = []
        line = features[i]
        values = eval(line)
        temp1 = temp.copy()
        for value in values:
            temp1[int(value[0])] = int(value[1])
        result.append(temp1)
    return pd.DataFrame(result)


def getdata(rawtraindata, rawvalidata, rawtestdata):
    train_x, train_y = rawtraindata['tweet'].copy(), rawtraindata[
        'region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = rawtestdata['tweet'].copy()
    validation_x, validation_y = rawvalidata['tweet'].copy(), rawvalidata['region'].copy()
    return train_x, train_y, validation_x, validation_y, test_x

def readVocab():
    loadVocab = []
    file = open('data/vocab.txt', 'r')
    for line in file.readlines():
        line = line.strip()
        key, value = line.split('\t')[0], line.split('\t')[1]
        loadVocab.append(key)
    file.close()
    return loadVocab

