import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':

    data_train_full=pd.read_csv('data/train_full.csv')
#data_train_count=pd.read_csv('data/train_count.csv')
#data_train_glove300=pd.read_csv('data/train_glove300.csv')
#data_train_tfidf=pd.read_csv('data/train_tfidf.csv')

    data_test_full=pd.read_csv('data/test_full.csv')
#data_test_count=pd.read_csv('data/test_count.csv')
#data_test_glove300=pd.read_csv('data/test_glove300.csv')
#data_test_tfidf=pd.read_csv('data/test_tfidf.csv')

    data_validation_full=pd.read_csv('data/dev_full.csv')
#data_validation_count=pd.read_csv('data/dev_count.csv')
#data_validation_glove300=pd.read_csv('data/dev_glove300.csv')
#data_validation_tfidf=pd.read_csv('data/dev_tfidf.csv')
    print("--------------------training data shape ------------------")
    print(data_train_full.shape)
    print('============================training data info============================')
    print(data_train_full.info())
    print("-----------------------------------------------")
    print(data_train_full.describe())
    #len(data_train_full.apply(lambda x: x.selected_text in x.text, axis=1))
    temp = data_train_full.groupby('region').count()['tweet'].reset_index().sort_values(by='tweet', ascending=False)
    print(temp)

    plt.figure(figsize=(12, 6))
    sns.countplot(x='region', data=data_train_full)
    plt.show()
    print("--------------------validation data shape ------------------")
    print(data_validation_full.shape)
    print('============================training data info============================')
    print(data_validation_full.info())
    print("-----------------------------------------------")
    print(data_validation_full.describe())
    # len(data_train_full.apply(lambda x: x.selected_text in x.text, axis=1))
    temp = data_validation_full.groupby('region').count()['tweet'].reset_index().sort_values(by='tweet', ascending=False)
    print(temp)

    plt.figure(figsize=(12, 6))
    sns.countplot(x='region', data=data_validation_full)
    plt.show()

    print("--------------------test data shape ------------------")
    print(data_test_full.shape)
    print('============================test data info============================')
    print(data_test_full.info())
    print("-----------------------------------------------")
    print(data_test_full.describe())
