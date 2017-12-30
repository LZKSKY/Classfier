from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib
import json
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_stc_text(file):
    texts = []
    in_file = open(file,'r')
    lines = in_file.readlines()
    in_file.close()
    doc_length = len(lines)
    for i in range(doc_length):
        text = (lines[i]).strip()
        texts.append(text)
    return texts

def read_stc_label(file):
    labels = []
    in_file = open(file,'r')
    lines = in_file.readlines()
    in_file.close()
    doc_length = len(lines)
    for i in range(doc_length):
        label = int((lines[i]).strip())
        labels.append(label)
    return labels

def run_use_stc(dataset):
    train_ratio_range = [0.1]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.1]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    # dataset = './dataset/Biomedical/'
    for i in range(len(train_ratio_range)):
        train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
        train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
        test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
        test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
        train_text = read_stc_text(train_text_file)
        train_label = read_stc_label(train_label_file)
        test_text = read_stc_text(test_text_file)
        test_label = read_stc_label(test_label_file)
        # count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
        # X_train_counts = count_vect.fit_transform(train_text)
        # tfidf_transformer = TfidfTransformer()
        # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # clf = MultinomialNB().fit(X_train_tfidf, train_label)
        text_clf = Pipeline([('vect', CountVectorizer(stop_words="english", decode_error='ignore')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        text_clf_2 = Pipeline([('vect', CountVectorizer(stop_words='english', decode_error='ignore')),
                               ('tfidf', TfidfTransformer()),
                               ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                     alpha=1e-3, max_iter = None, tol = 1e-3, random_state=42)),
                               ])
        text_clf = text_clf.fit(train_text, train_label)
        predicted = text_clf.predict(test_text)
        text_clf_2 = text_clf_2.fit(train_text, train_label)
        predicted_2 = text_clf_2.predict(test_text)
        acc.append([np.mean(predicted == test_label),np.mean(predicted_2 == test_label)])
        # acc.append([np.mean(predicted == test_label),np.mean(predicted_2 == test_label)])
    print(acc)
    # x = [i/10.0 for i in range(1,len(train_ratio_range))]
    '''
    x = [0.01]
    style = ['r*','g*','b*']
    for i in range(len(acc)):
        plt.plot(x,acc[i],style[i])
    plt.show()
    '''

def read_newTweets(file):
    texts = []
    labels = []
    in_file = open(file,'r')
    for line in in_file:
        obj = json.loads(line)
        texts.append(obj['textCleaned'])
        labels.append(obj['clusterNo'])
    in_file.close()
    return (texts, labels)

def run_newTweets(dataset):
    train_ratio_range = [0.1]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.1]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    # dataset = './dataset/Biomedical/'
    for i in range(len(train_ratio_range)):
        train_file = dataset + 'train' + str(train_ratio_range[i])
        test_file = dataset + 'test' + str(test_ratio_range[i])
        (train_text,train_label) = read_newTweets(train_file)
        (test_text,test_label) = read_newTweets(test_file)
        # count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
        # X_train_counts = count_vect.fit_transform(train_text)
        # tfidf_transformer = TfidfTransformer()
        # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # clf = MultinomialNB().fit(X_train_tfidf, train_label)
        text_clf = Pipeline([('vect', CountVectorizer(stop_words="english", decode_error='ignore')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        text_clf_2 = Pipeline([('vect', CountVectorizer(stop_words='english', decode_error='ignore')),
                               ('tfidf', TfidfTransformer()),
                               ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                     alpha=1e-3, max_iter=None, tol=1e-3, random_state=42)),
                               ])
        text_clf = text_clf.fit(train_text, train_label)
        predicted = text_clf.predict(test_text)
        text_clf_2 = text_clf_2.fit(train_text, train_label)
        predicted_2 = text_clf_2.predict(test_text)
        acc.append([np.mean(predicted == test_label), np.mean(predicted_2 == test_label)])
        # acc.append([np.mean(predicted == test_label),np.mean(predicted_2 == test_label)])
    print(acc)

path = './dataset/'
dataset = ['Biomedical/','SearchSnippets/','StackOverflow/']



if __name__ == '__main__':
    run_newTweets('data1/')
    # for i in range(0,len(dataset)):
    #     run_use_stc(path + dataset[i])