import json
import Document

class classifier:
    def __init__(self, V, wordList):
        self.z = {}
        self.m_z = {}
        self.n_z = {}
        self.n_zv = {}
        self.iterNum = 2
        self.ClusterNum = 0
        self.D = 0
        self.alpha = 40
        self.beta = 0.08
    def train(self, documents):
        for document in documents:
            documentID = document.documentID
            cluster = document.clusterID
            self.z[documentID] = cluster
            self.ClusterNum = max(self.ClusterNum, cluster + 1)
            if cluster not in self.m_z:
                self.m_z[cluster] = 0
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if cluster not in self.n_zv:
                    self.n_zv[cluster] = {}
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre
                if cluster not in self.n_z:
                    self.n_z[cluster] = 0
                self.n_z[cluster] += wordFre
            self.D += 1

    def assign_labels(self, documents):
        for document in documents:
            self.D += 1
            documentID = document.documentID
            cluster = self.sampleCluster(document)
            self.z[documentID] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

    def predict(self, documents):
        self.assign_labels(documents)
        for i in range(self.iterNum):
            for document in documents:
                documentID = document.documentID
                cluster = self.z[documentID]
                self.ClusterNum = max(self.ClusterNum, cluster + 1)
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                cluster = self.sampleCluster(document)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

    def cal_accuracy(self, documents):
        true_num = 0
        all_num = 0
        for document in documents:
            all_num += 1
            documentID = document.documentID
            cluster_pred = self.z[documentID]
            cluster_label = document.clusterID
            if cluster_label == cluster_pred:
                true_num += 1
        return true_num/float(all_num)



    def sampleCluster(self, document):
        prob = [float(0.0)] * self.ClusterNum
        for cluster in range(self.ClusterNum):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        kChoosed = 0
        k_max = prob[0]
        for k in range(1, self.ClusterNum):
            if k_max < prob[k]:
                kChoosed = k
                k_max = prob[k]
        return kChoosed