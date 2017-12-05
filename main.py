import json
from Document import Document
from classifier import classifier
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

    # output_test(documentSet, outputPath, wordList)


def runSimpleClassifier():
    train_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = []
    for i in range(len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_file = './data1/train' + str(train_ratio_range[i])
        test_file = './data1/test' + str(test_ratio_range[i])
        train_doc = readfile(train_file, wordToIdMap, wordList)
        test_doc = readfile(test_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList)
        classify.train(train_doc)
        classify.predict(test_doc)
        tmp = classify.cal_accuracy(test_doc)
        acc.append(tmp)
    print(acc)
    plt.plot(train_ratio_range, acc, 'r-')
    plt.show()

def testlast_ten():

    train_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]
    test_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]
    for i in range(5, len(train_ratio_range)):
        wordList = []
        wordToIdMap = {}
        train_file = './data/train' + str(train_ratio_range[i])
        test_file = './data/train' + str(test_ratio_range[i])
        train_doc = readfile(train_file, wordToIdMap, wordList)
        test_doc = readfile(test_file, wordToIdMap, wordList)
        classify = classifier(len(wordToIdMap), wordList)
        classify.train(train_doc)
        classify.predict(test_doc)
        acc = classify.cal_accuracy(test_doc)
        print(str(train_ratio_range[i]) + ' ' + str(acc))


if __name__ == '__main__':
    runSimpleClassifier()
    # testlast_ten()
