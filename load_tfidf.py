import numpy as np
import pandas as pd

def read_data():
    print("loading data......")
    data_train_tfidf = pd.read_csv('data/train_tfidf.csv')
    data_test_tfidf = pd.read_csv('data/test_tfidf.csv')
    data_vali_tfidf = pd.read_csv('data/dev_tfidf.csv')
    train_x, train_y = data_train_tfidf['tweet'].copy(), data_train_tfidf[
        'region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = data_test_tfidf['tweet'].copy()
    validation_x, validation_y = data_vali_tfidf['tweet'].copy(), data_vali_tfidf['region'].copy()
    max_len = max(
        [len(eval(x)) for x in train_x] + [len(eval(x)) for x in validation_x] + [len(eval(x)) for x in test_x])

    train_x = preprocesdata_tfidf_maxCreates(train_x, max_len)
    test_x = preprocesdata_tfidf_maxCreates(test_x, max_len)
    validation_x = preprocesdata_tfidf_maxCreates(validation_x, max_len)
    print("training data: ")
    print(train_x)
    print("training labels: ")
    print(train_y)
    return train_x,train_y,validation_x,validation_y, test_x


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