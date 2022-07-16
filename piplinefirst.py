import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import csv
resultPath = 'results'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# with

def getdata(rawtraindata, rawvalidata, rawtestdata):
    train_x, train_y = rawtraindata['tweet'].copy(), rawtraindata['region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = rawtestdata['tweet'].copy()
    validation_x, validation_y = rawvalidata['tweet'].copy(), rawvalidata['region'].copy()
    return train_x,train_y,validation_x,validation_y, test_x

def saveResult(data):
    df = pd.DataFrame()
    df['region'] = data
    df['id'] = df.index + 1
    df.to_csv(resultPath+'/result1.csv',index=None)


if __name__ == '__main__':
    data_train_glove300 = pd.read_csv('data/train_glove300.csv')
    data_test_glove300 = pd.read_csv('data/test_glove300.csv')
    data_validation_glove300 = pd.read_csv('data/dev_glove300.csv')
    user_infoData = pd.read_table('data/rawdata/GeoText.2010-10-12/processed_data/user_info',sep='\t')
    print(user_infoData.head())
    train_x_y = pd.merge(data_train_glove300, user_infoData, how='left', on=['user'])
    vali_x_y = pd.merge(data_validation_glove300, user_infoData, how='left', on=['user'])
    test_x_y = pd.merge(data_test_glove300, user_infoData, how='left', on=['user'])
    train_x_y.drop('user', inplace=True, axis=1)
    test_x_y.drop('user', inplace=True, axis=1)
    vali_x_y.drop('user', inplace=True, axis=1)
    train_x_y.drop('tweet', inplace=True, axis=1)
    test_x_y.drop('tweet', inplace=True, axis=1)
    vali_x_y.drop('tweet', inplace=True, axis=1)
    tweet_list = []
    for i in range(len(data_train_glove300['tweet'].copy()[0].split(' '))):
        temp = 'tweet' + str(i)
        tweet_list.append(temp)
    train_x_300 = pd.DataFrame()
    train_x_300[tweet_list] = data_train_glove300['tweet'].str.split(pat=" ", expand=True).astype('float64')
    test_x_300 = pd.DataFrame()
    test_x_300[tweet_list] = data_test_glove300['tweet'].str.split(pat=" ", expand=True).astype('float64')
    vali_x_300 = pd.DataFrame()
    vali_x_300[tweet_list] = data_validation_glove300['tweet'].str.split(pat=" ", expand=True).astype('float64')


    train_x_no300 = train_x_y.drop('region', inplace=False, axis=1)
    test_x_no300 = test_x_y.drop('region', inplace=False, axis=1)
    vali_x_no300 = vali_x_y.drop('region', inplace=False, axis=1)
    train_x,train_y = pd.concat([train_x_no300,train_x_300], axis=1),train_x_y['region'].copy()
    test_x, test_y = pd.concat([test_x_no300,test_x_300], axis=1), test_x_y['region'].copy()
    vali_x, vali_y = pd.concat([vali_x_no300,vali_x_300], axis=1), vali_x_y['region'].copy()

    classifier_train = GaussianNB()
    classifier_train.fit(train_x, train_y)
    classifier_predict_validation = classifier_train.predict(vali_x)

    print(classifier_predict_validation)
    print(accuracy_score(vali_y, classifier_predict_validation))
    print(f1_score(vali_y, classifier_predict_validation, average='macro'))

    saveResult(classifier_train.predict(test_x))

