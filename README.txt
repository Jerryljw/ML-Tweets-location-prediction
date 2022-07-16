EDA.py----The exploration on  raw dataset
load_tfidf.py----load tfidf dataset and use a WF--embedding word codes and tfidf value to same length input vector
load_wordcount.py----load word count dataset and use a WF--embedding word codes and count value to same length input vector
plts.py---plot the grid search plots
preprocess_tfidf.py----load tfidf dataset and use a WC--make each word to a feature column, take tfidf value as value
preprocess_wordcount.py----load tfidf dataset and use a WC--make each word to a feature column, take tfidf value as value

result output methods::
tfidf_functionsOutput.py----various ML method output using WF dataset and tfidf
tfidf_coded.py----various ML method output using WC dataset and tfidf
tfidf_tensorflow.py----DNN method output using WF dataset and tfidf
tfidf_tensorflow_coded.py----DNN method output using WC dataset and tfidf

word_count_func.py----various ML method output using WF dataset and word count
word_count_coded.py----various ML method output using WC dataset and word count
word_tensorflow.py----DNN method output using WF dataset and word count
word_tensorflow_coded.py----DNN method output using WC dataset and word count

To get evaluation matrics, run result output methods; to get graphs, run EDA and uncommon the grid search plot on tfidf_functionsOutput.py