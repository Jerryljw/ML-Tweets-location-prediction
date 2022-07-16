import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import pandas as pd

def plotGDs(gsCV, gs_param):
    cvres = gsCV.cv_results_
    knn_k = gs_param['n_neighbors']
    test_scores = cvres['mean_test_score']
    test_scores = np.array(test_scores).reshape(len(knn_k), 1)
    _, pic = plt.subplots(1, 1)
    pic.plot(knn_k, test_scores, '-o')
    pic.set_title("Grid Search Plot")
    pic.set_xlabel("k value")
    pic.set_ylabel('f1-score')
    pic.legend(loc="best")
    pic.grid('on')
    plt.show()