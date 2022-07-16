import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import time
from sklearn.dummy import DummyClassifier
import load_tfidf
import load_wordcount

resultPath = 'results'


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def saveResult(data,name):
    df = pd.DataFrame()
    df['region'] = data
    df['id'] = df.index + 1
    df.to_csv(resultPath+'/wordCount_func_result'+name +'.csv',index=None)


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



def produce_result(msg, classifier, train_x, train_y, validation_x,validation_y,test_x):
    print("the performance matrix of "+msg + " : ")
    begin = time.time()
    print("the program start run at " + begin.__str__())
    classifier.fit(train_x, train_y)
    classifier_predict_validation = classifier.predict(validation_x)
    end = time.time()
    print("the program end run at " + end.__str__() + " , used: " + str(end - begin)+ " seconds")
    performanceMatrix(validation_y, classifier_predict_validation)
    test_predict = classifier.predict(test_x)
    saveResult(test_predict,msg)


if __name__ == '__main__':

    train_x,train_y,validation_x,validation_y, test_x = load_wordcount.read_data()
    produce_result("Zero_R", DummyClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("naive_bayes", BernoulliNB(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("neural_network", MLPClassifier(learning_rate_init=0.0001, early_stopping=True), train_x, train_y,
                   validation_x, validation_y, test_x)
    produce_result("k_nearest_neighbour", KNeighborsClassifier(n_neighbors=5), train_x, train_y, validation_x,
                   validation_y, test_x)

    produce_result("Stochastic_Gradient_Descent", SGDClassifier(), train_x, train_y, validation_x, validation_y, test_x)
    produce_result("AdaBoost", AdaBoostClassifier(n_estimators=140, learning_rate=1), train_x, train_y, validation_x,
                   validation_y, test_x)
    produce_result("Random_Forest", RandomForestClassifier(max_depth=30, n_estimators=140), train_x, train_y,
                   validation_x,
                   validation_y, test_x)





