import json
from Document import Document
from classifier import classifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def readfile(filename, wordToIdMap, wordList):
    documents = []
    with open(filename) as input:
        line = input.readline()
        while line:
            obj = json.loads(line)
            text = obj['textCleaned']
            clusterID = int(obj['clusterNo'])
            document = Document(text, wordToIdMap, wordList, int(obj['tweetId']), clusterID)
            documents.append(document)
            line = input.readline()
    return documents

def read_stc(text_file, label_file, wordToIdMap, wordList):
    documents = []
    doc_text_file = open(text_file,'r')
    doc_label_file = open(label_file,'r')
    doc_text_lines = doc_text_file.readlines()
    doc_label_lines = doc_label_file.readlines()
    doc_text_file.close()
    doc_label_file.close()
    if len(doc_text_lines) != len(doc_label_lines):
        print('word and label file have not same length')
        exit(-1)
    doc_length = len(doc_text_lines)
    for i in range(doc_length):
        text = doc_text_lines[i]
        label = int(doc_label_lines[i].strip())
        document = Document(text, wordToIdMap, wordList, i, label)
        documents.append(document)
    return documents

def run(beta, prob):
    train_ratio_range = [0.1]
    test_ratio_range = [0.1]
    acc = []
    for i in range(len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_file = './data1/train' + str(train_ratio_range[i])
        test_file = './data1/test' + str(test_ratio_range[i])
        train_doc = readfile(train_file, wordToIdMap, wordList)
        test_doc = readfile(test_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList,beta, 0)
        classify.train(train_doc)
        classify.predict_with_prob_10(test_doc)
        tmp = classify.cal_accuracy(test_doc)
        acc.append(tmp)
    print(acc)
    # plt.plot(train_ratio_range, acc, 'r-')
    # plt.show()

def run_use_stc(dataset, beta, prob):
    train_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    for i in range(len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
        train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
        test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
        test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
        train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
        test_doc = read_stc(test_text_file,test_label_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList, beta, 0)
        classify.train(train_doc)
        classify.predict(test_doc)
        tmp = classify.cal_accuracy(test_doc)
        acc.append(tmp)
    print(acc)

def run_use_stc_with_beta_range(dataset, prob):
    train_ratio_range = [0.1]
    test_ratio_range = [0.1]
    beta_range = [i/100.0 for i in range(1,10)] + [i/10.0 for i in range(1,10)] + [i for i in range(1,10)]
    acc = []
    acc_beta_range = []
    for beta in beta_range:
        for i in range(len(train_ratio_range)):
            wordList = []
            wordToIdMap = {}
            train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
            train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
            test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
            test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
            train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
            test_doc = read_stc(test_text_file,test_label_file, wordToIdMap, wordList)
            classify = classifier(len(wordToIdMap), wordList, beta, 0.8)
            classify.train(train_doc)
            classify.predict_with_prob_10(test_doc)
            tmp = classify.cal_accuracy(test_doc)
            acc.append(tmp)
        acc_beta_range.append(acc)
        acc = []
    print(acc_beta_range)

def run_use_stc_with_V_change_prob_increment(dataset, prob):
    train_ratio_range = [0.1]
    test_ratio_range = [0.1]
    beta_range = [i/100.0 for i in range(1,10)] + [i/10.0 for i in range(1,10)] + [i for i in range(1,10)]
    acc = []
    acc_beta_range = []
    for beta in beta_range:
        for i in range(len(train_ratio_range)):
            wordList = []
            wordToIdMap = {}
            train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
            train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
            test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
            test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
            train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
            test_doc = read_stc(test_text_file,test_label_file, wordToIdMap, wordList)
            classify = classifier(len(wordToIdMap), wordList, beta, 0.8)
            classify.train_with_V_change(train_doc)
            classify.assign_labels_with_prob_10_V(test_doc)
            tmp = classify.cal_accuracy(test_doc)
            acc.append(tmp)
        acc_beta_range.append(acc)
        acc = []
    print(acc_beta_range)

def run_use_stc_with_prob_range(dataset,beta):
    train_ratio_range = [0.1]
    test_ratio_range = [0.1]
    prob_add_range = [0.8]
    acc = []
    acc_probadd_range = []
    for prob_add in prob_add_range:
        for i in range(len(train_ratio_range)):
            wordList = []
            wordToIdMap = {}
            train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
            train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
            test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
            test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
            train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
            test_doc = read_stc(test_text_file, test_label_file, wordToIdMap, wordList)
            classify = classifier(len(wordToIdMap), wordList, beta, prob_add)
            classify.train(train_doc)
            classify.predict_with_prob_10(test_doc)
            tmp = classify.cal_accuracy(test_doc)
            acc.append(tmp)
        print(acc)
        acc_probadd_range.append(acc)
        acc = []
    print(acc_probadd_range)

def run_use_stc_with_prob_raw(dataset,beta):
    train_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    for i in range(len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
        train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
        test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
        test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
        train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
        test_doc = read_stc(test_text_file, test_label_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList, beta, 0)
        classify.train(train_doc)
        classify.predict_with_prob_raw(test_doc)
        tmp = classify.cal_accuracy(test_doc)
        acc.append(tmp)
    print(acc)

def run_use_stc_with_V_change(dataset,beta):
    train_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    for i in range(len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_text_file = dataset + 'train_text' + str(train_ratio_range[i]) + '.txt'
        train_label_file = dataset + 'train_label' + str(train_ratio_range[i]) + '.txt'
        test_text_file = dataset + 'test_text' + str(test_ratio_range[i]) + '.txt'
        test_label_file = dataset + 'test_label' + str(test_ratio_range[i]) + '.txt'
        train_doc = read_stc(train_text_file, train_label_file, wordToIdMap, wordList)
        test_doc = read_stc(test_text_file, test_label_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList, beta, 0)
        classify.train_with_V_change(train_doc)
        classify.predict_with_V_change(test_doc)
        tmp = classify.cal_accuracy(test_doc)
        acc.append(tmp)
        print(acc)
    print(acc)

path = './dataset/'
dataset = ['Biomedical/','SearchSnippets/','StackOverflow/']
beta = 0.3
prob_add = 0.8
if __name__ == '__main__':
    beta_range = [i/100.0 for i in range(1,10)] + [i/10.0 for i in range(1,10)] + [i for i in range(1,10)]
    for beta in beta_range:
        run(beta, prob_add)
    for i in range(len(dataset)):
        # run_use_stc(path + dataset[i], beta, prob_add)
        # run_use_stc_with_prob_range(path + dataset[i], beta)
        # run_use_stc_with_prob_raw(path + dataset[i], beta)
        # run_use_stc_with_beta_range(path + dataset[i], prob_add)
        # run_use_stc_with_V_change(path + dataset[i], beta)
        # run_use_stc_with_V_change_prob_increment(path + dataset[i], prob_add)
        pass