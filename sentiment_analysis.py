# Author: Dimosthenis Antypas

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
import nltk
import re
import itertools
from joblib import Parallel,delayed
from multiprocessing import cpu_count
import time
import os
import sys


# reviews filenames
train_pos = "./IMDb/train/imdb_train_pos.txt"
train_neg = "./IMDb/train/imdb_train_neg.txt"
dev_pos = "./IMDb/dev/imdb_dev_pos.txt"
dev_neg = "./IMDb/dev/imdb_dev_neg.txt"
test_pos = "./IMDb/test/imdb_test_pos.txt"
test_neg = "./IMDb/test/imdb_test_neg.txt"

# lexicons filenames
words_pos = "./opinion-lexicon-English/positive-words.txt"
words_neg = "./opinion-lexicon-English/negative-words.txt"

# if needed uncomment
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")


def getSet(positive, negative, printLength=False):
    """Function to combine positive and negative reviews and label
        them 1 for positive and 0 for negative. If printLength is True
        the number of positive and negative Reviews in each set is printed.
    """
    with open(positive, encoding='ISO8859-1') as file:
        lines_pos = file.readlines()
        if printLength: print("Positive Reviews: ", len(lines_pos))

    with open(negative, encoding='ISO8859-1') as file:
        lines_neg = file.readlines()
        if printLength: print("Negative Reviews: ", len(lines_neg))

    s = []
    for line in lines_pos:
        # remove html tags
        line = re.sub('<[^<]+?>', '', line)
        s.append((line, 1))
    for line in lines_neg:
        # remove html tags
        line = re.sub('<[^<]+?>', '', line)
        s.append((line, 0))

    return s


def readLexicon(filename):
    """Function to read lexicon"""

    with open(filename, encoding="ISO-8859-1") as file:
        while file.readline()[0] == ";":
            file.readline()
        lines = file.readlines()

    result = [x.split("\n")[0] for x in lines]
    return result


def get_list_tokens(string,mode):
    """ Tokenize given string
    Mode = 1 Tokenize and get terms that only appear in the lexicon.
    Mode = 2 Remove stopwords and short words (len(word) < 3). Then tokenize and get terms.
    Mode = 3 Tokenize based on part of speech. Use terms that are adjectives or verbs.
    Mode = 4 Tokenize based on part of speech and lexicon. Use terms that are adjectives or verbs or that appear in the lexicon """

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenized_sentence = nltk.tokenize.word_tokenize(string)
    list_tokens = []
    if (mode == 1):
        for token in tokenized_sentence:
            if token.lower() in lexicon:
                list_tokens.append(lemmatizer.lemmatize(token).lower())
    elif (mode == 2):
        for token in tokenized_sentence:
            if (token not in stopwords) and (len(token) > 3):
                list_tokens.append(lemmatizer.lemmatize(token).lower())
    elif (mode == 3):
        for token, position in nltk.pos_tag(tokenized_sentence):
            if position in ["VB", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                list_tokens.append(lemmatizer.lemmatize(token).lower())
    elif (mode == 4):
        for token, position in nltk.pos_tag(tokenized_sentence):
            if position in ["VB", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] or token.lower() in lexicon:
                list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


def worker(option):
    """Worker function which for each combination of feature selection options, creates a TfidfVectorizer, fits 
    the train set in it and then uses the vocabulary created it to transform each text review to a vector. Then using the 
    SelectKBest and chi2 functions the 250 most useful features are kept and used to fit the SVM model. Dev set is then used
    to evaluate the model and the results are returned."""
    ngram = option[0]
    tokenization_option = option[1]

    # create a TfidfVectorizer with given ngram and get_list_tokens as tokenizer
    tf = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=ngram,
                         tokenizer=lambda text: get_list_tokens(text, tokenization_option))
    
    train_corpus = [instance[0] for instance in train_set]
    tfidf_matrix = tf.fit_transform(train_corpus)
    X_train = tfidf_matrix
    y_train = [instance[1] for instance in train_set]

    fs_sentanalysis = SelectKBest(chi2, k=250).fit(X_train, y_train)
    X_train_new = fs_sentanalysis.transform(X_train)

    dev_corpus = [instance[0] for instance in dev_set]
    X_dev = tf.transform(dev_corpus)
    y_dev = [instance[1] for instance in dev_set]

    X_dev_new = fs_sentanalysis.transform(X_dev)

    model = SVC(kernel='linear')
    model.fit(X_train_new, y_train)
    p = model.predict(X_dev_new)

    accuracy = accuracy_score(p, y_dev)
    precision = precision_score(p, y_dev)
    recall = recall_score(p, y_dev)
    f1 = f1_score(p, y_dev)

    """results list contain a tuple of the model related objects (model,tfidfVectorizer,SelectKBest), 
    the ngram used, the type of tokenization used and a number of performance meters (accuracy,precision,recall,f1)"""
    if tokenization_option == 1:
        results = [(model, tf, fs_sentanalysis), ngram,
                   'lexicon', accuracy, precision, recall, f1]
    elif tokenization_option == 2:
        results = [(model, tf, fs_sentanalysis), ngram, 'stopwords and len(word)>3',
                   accuracy, precision, recall, f1]
    elif tokenization_option == 3:
        results = [(model, tf, fs_sentanalysis), ngram,
                   'pos', accuracy, precision, recall, f1]
    elif tokenization_option == 4:
        results = [(model, tf, fs_sentanalysis), ngram,
                   'pos or lexicon', accuracy, precision, recall, f1]
    return results


# create lexicon
pos_lexicon = readLexicon(words_pos)
neg_lexicon = readLexicon(words_neg)
lexicon = pos_lexicon + neg_lexicon
lexicon = set(lexicon)

# get train,dev,test sets
train_set = getSet(train_pos, train_neg,True)
dev_set = getSet(dev_pos, dev_neg)
test_set = getSet(test_pos, test_neg)


# feature selection options to be tested (ngram option and tokenization mode for get_list_tokens())
parameters = [[(1, 1), (2, 2)], [1,2,3,4]]
# get possible combinations
parameter_list = list(itertools.product(parameters[0], parameters[1]))

# User input for multiprocessing options.
# User can add an extra integer argument when running the script which will indicate the number of cpu cores to be used (no input = 1 core).
user_input = sys.argv
t1 = time.time()


# select sequential/parallel execution based on user input.
if len(user_input) == 1:
    if __name__ == '__main__':
        results = Parallel(n_jobs=1)(delayed(worker)(option) for option in parameter_list)
else:
    try:
        jobs = int(user_input[1])
        if (jobs > cpu_count()):
            jobs = cpu_count()

        if __name__ == '__main__':
            results = Parallel(n_jobs=jobs)(delayed(worker)(option) for option in parameter_list)
    except ValueError:
       raise SystemExit('Invalid input')

# get results in a dataframe
df = pd.DataFrame(results, columns=[
                  "model", "ngram", "tokenization_option", "accuracy", "precision", "recall", "f1"])


# print sorted results and get best options based on accuracy
print(df.sort_values(by='accuracy', ascending=False).iloc[:, 1:])
best_options = df.sort_values(by='accuracy', ascending=False).iloc[0]
best_ngram = best_options["ngram"]
best_tokenization_option = best_options["tokenization_option"]
best_model = best_options["model"][0]
best_tf = best_options["model"][1]
best_fs = best_options["model"][2]
print("\nBest options: \n\t Ngram: {} \n\t tokenization_option: {}".format(
    best_ngram, best_tokenization_option))

# test the best model using the test set
test_corpus = [instance[0] for instance in test_set]
X_test = best_tf.transform(test_corpus)
y_test = [instance[1] for instance in test_set]

X_test_new = best_fs.transform(X_test)
print("Number of features before feature selection",len(best_tf.get_feature_names()))
print("Number of features after feture selection", 250)

predictions = best_model.predict(X_test_new)
print("\nClassification report on the test set:\n")
print(classification_report(predictions,y_test))

t2 = time.time()
total_time = (t2-t1)/60
print("\nTotal time: ", round(total_time, 3), " minutes")
