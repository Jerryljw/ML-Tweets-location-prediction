import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def read_data():
    print("loading data......")
    data_train_glove300 = pd.read_csv('data/train_glove300.csv')

    data_test_glove300 = pd.read_csv('data/test_glove300.csv')

    data_validation_glove300 = pd.read_csv('data/dev_glove300.csv')
    train_x, train_y = data_train_glove300['tweet'].copy(), data_train_glove300[
        'region'].copy()  # data_train_glove300.iloc[:, [1,2]].copy(),data_train_glove300['region'].copy()
    test_x = data_test_glove300['tweet'].copy()
    validation_x, validation_y = data_validation_glove300['tweet'].copy(), data_validation_glove300['region'].copy()
    tweet_list = []
    for i in range(len(train_x[0].split(' '))):
        temp = 'tweet' + str(i)
        tweet_list.append(temp)
    train_x_300 = pd.DataFrame()
    train_x_300[tweet_list] = train_x.str.split(pat=" ", expand=True).astype('float64')
    # train_x_300.drop('tweet', axis=1,inplace = True)

    validation_x_300 = pd.DataFrame()
    validation_x_300[tweet_list] = validation_x.str.split(pat=" ", expand=True).astype('float64')
    # validation_x_300.drop('tweet', axis=1,inplace = True)

    test_x_300 = pd.DataFrame()
    test_x_300[tweet_list] = test_x.str.split(pat=" ", expand=True).astype('float64')
    # test_x_300.drop('tweet', axis=1,inplace = True)
    return train_x_300, train_y,  validation_x_300, validation_y,test_x_300
