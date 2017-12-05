import codecs
import json
from sklearn import metrics
import pylab as py
import numpy as np
'''
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
'''
class ClusterEvaluation():
    def __init__(self):
        self.labelsPred = {}
        self.labelsTrue = {}
        self.labelsPredAndTrue = []

    def getMStreamPredLabels(self, inFile):
        with codecs.open(inFile, 'r') as fin:
            for line in fin:
                try:
                    documentID = line.strip().split()[0]
                    clusterNo = line.strip().split()[1]
                    self.labelsPred[int(documentID)] = int(clusterNo)
                except:
                    print(line)

    def getMStreamTrueLabels(self, inFile, dataset):
        self.labelsTrue = {}
        self.docs = []
        self.labelsPredAndTrue = []
        outFile = inFile + "Full.txt"
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue[int(docObj['tweetId'])] = int(docObj['clusterNo'])
                    self.docs.append([int(docObj['tweetId']), docObj['textCleaned']])
                except:
                    print(docJson)

        with codecs.open(outFile, 'w') as fout:
            counter = 0
            num_all = 0
            for i in range(len(self.docs)):
                docObj = {}
                documentID = self.docs[i][0]
                if documentID in self.labelsPred:
                    docObj['trueCluster'] = self.labelsTrue[documentID]
                    docObj['predictedCluster'] = self.labelsPred[documentID]
                    self.labelsPredAndTrue.append([self.labelsTrue[documentID], self.labelsPred[documentID]])
                    num_all += 1
                    if self.labelsTrue[documentID] == self.labelsPred[documentID]:
                        counter += 1
                    docObj['text'] = self.docs[i][1]
                    # docObj['clusterName'] = clusterNames[i]
                    docJson = json.dumps(docObj)
                    fout.write(docJson + '\n')
            # print(str(num_all) + '**' + str(counter))
            # print(str(metrics.accuracy_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue])))

    def get_classification_report(self):
        return metrics.classification_report([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue])
    def get_accuracy_score(self):
        return metrics.accuracy_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue])
    def get_confusion_matrix(self):
        return metrics.confusion_matrix([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue])
    def get_average_accuracy(self):
        return np.mean(metrics.precision_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue], average='micro'))
    def get_f1_score(self):
        return metrics.f1_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue], average='micro')
    def get_precision(self):
        return metrics.precision_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue], average='micro')
    def get_recall(self):
        return metrics.recall_score([i[0] for i in self.labelsPredAndTrue], [i[1] for i in self.labelsPredAndTrue], average='micro')



def draw_pic(x, accuracy_score_set, average_accuracy_set):
    Xlabel = 'ratioRange'
    Ylabel = 'performance'
    titleStr = 'classify result'
    py.figure()
    p1 = py.plot(x, accuracy_score_set, 'r-*')
    p2 = py.plot(x, average_accuracy_set, 'g-D')
    py.legend([p1[0], p2[0], ['accuracy_score', 'average_accuracy']])
    # py.errorbar(XRange, KPredNumMeanList, yerr=KPredNumVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()


def runWithRatio():
    K = 0
    iterNum = 2
    ratioRange = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    classify_report_set = []
    confusion_matrix_set = []
    accuracy_score_set = []
    precision_set = []
    recall_set = []
    average_accuracy_set = []
    f1_score_set = []
    sampleNum = 1
    alpha = '40'
    beta = '0.02'
    KThreshold = 0
    dataset = 'newtweets'
    datasetPath = './data/' + dataset
    inPath = './result/'
    resultFileName = 'MStreamNoiseKThreshold%dIterNumDataset%sK%dsampleNum%dalpha%sbeta%sIterNum%s.txt' % (KThreshold,
                                                                                                           dataset, K,
                                                                                                           sampleNum,
                                                                                                           alpha, beta,
                                                                                                           iterNum)
    resultFilePath = './result_eval/' + resultFileName
    MStreamEvaluation = ClusterEvaluation()
    for ratio in ratioRange:
        for sampleNo in range(1, sampleNum + 1):
            MStreamEvaluation.labelsPred = {}
            dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%s%s/' % \
                      (dataset, K, iterNum, sampleNum, alpha, beta, ratio)
            inDir = inPath + dirName
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamPredLabels(inFile)
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNum" + str(sampleNum) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            # classify_report = MStreamEvaluation.get_classification_report()
            # confusion_matrix = MStreamEvaluation.get_confusion_matrix()
            accuracy_score = MStreamEvaluation.get_accuracy_score()
            # average_accuracy = MStreamEvaluation.get_average_accuracy()
            # f1_score = MStreamEvaluation.get_f1_score()
            # precision = MStreamEvaluation.get_precision()
            recall = MStreamEvaluation.get_recall()
            # classify_report_set.append(classify_report)
            # confusion_matrix_set.append(confusion_matrix)
            accuracy_score_set.append(accuracy_score)
            recall_set.append(recall)
            # average_accuracy_set.append(average_accuracy)
            # f1_score_set.append(f1_score)
            # precision_set.append(precision)

    print(accuracy_score_set)
    # print(average_accuracy_set)
    # print(f1_score_set)
    # print(precision_set)
    # print(recall_set)
    # draw_pic(ratioRange, accuracy_score_set, average_accuracy_set)



if __name__ == '__main__':
    runWithRatio()