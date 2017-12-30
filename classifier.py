import json
import Document

class classifier:
    def __init__(self, V, wordList, beta, prob_add):
        self.z = {}
        self.m_z = {}
        self.n_z = {}
        self.n_zv = {}
        self.iterNum = 0
        self.ClusterNum = 0
        self.D = 0
        self.alpha = 10
        self.beta = beta
        self.V_beta = beta
        self.IsWordappear = [0]*(V + 1)
        self.V = 0
        self.prob_on = {}
        self.prob_add = prob_add

    def train(self, documents):
        self.V = len(self.IsWordappear) - 1
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

    def train_with_V_change(self, documents):
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
                if self.IsWordappear[wordNo] == 0 and wordFre > 0:
                    self.IsWordappear[wordNo] = 1
                    self.V += 1
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
            (cluster,prob) = self.sampleCluster(document)
            self.z[documentID] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

    def assign_labels_with_V_change(self, documents):
        for document in documents:
            self.D += 1
            documentID = document.documentID
            (cluster,prob) = self.sampleCluster(document)
            self.z[documentID] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if self.IsWordappear[wordNo] == 0 and wordFre > 0:
                    self.IsWordappear[wordNo] = 1
                    self.V += 1
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

    def assign_labels_with_prob_10(self, documents):
        for document in documents:
            self.D += 1
            documentID = document.documentID
            (cluster,prob) = self.sampleCluster(document)
            if prob >= self.prob_add:
                self.prob_on[documentID] = 1
            else:
                self.prob_on[documentID] = 0
            self.z[documentID] = cluster
            self.m_z[cluster] += self.prob_on[documentID]
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre * self.prob_on[documentID]
                self.n_z[cluster] += wordFre * self.prob_on[documentID]

    def assign_labels_with_prob_10_V(self, documents):
        for document in documents:
            self.D += 1
            documentID = document.documentID
            (cluster,prob) = self.sampleCluster(document)
            if prob >= self.prob_add:
                self.prob_on[documentID] = 1
            else:
                self.prob_on[documentID] = 0
            self.z[documentID] = cluster
            self.m_z[cluster] += self.prob_on[documentID]
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if self.IsWordappear[wordNo] == 0 and wordFre > 0:
                    self.IsWordappear[wordNo] = 1
                    self.V += 1
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre * self.prob_on[documentID]
                self.n_z[cluster] += wordFre * self.prob_on[documentID]
        # print(str(self.V) + "#"+str(len(self.IsWordappear)))

    def assign_labels_with_prob_raw(self, documents):
        for document in documents:
            self.D += 1
            documentID = document.documentID
            (cluster,prob) = self.sampleCluster(document)
            self.prob_on[documentID] = prob
            self.z[documentID] = cluster
            self.m_z[cluster] += self.prob_on[documentID]
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre * self.prob_on[documentID]
                self.n_z[cluster] += wordFre * self.prob_on[documentID]

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
                (cluster, prob) = self.sampleCluster(document)
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

    def predict_with_V_change(self, documents):
        self.assign_labels_with_V_change(documents)
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
                (cluster, prob) = self.sampleCluster(document)
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

    def predict_with_prob_10(self, documents):
        self.assign_labels_with_prob_10(documents)
        for i in range(self.iterNum):
            for document in documents:
                documentID = document.documentID
                cluster = self.z[documentID]
                self.ClusterNum = max(self.ClusterNum, cluster + 1)
                self.m_z[cluster] -= self.prob_on[documentID]
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre * self.prob_on[documentID]
                    self.n_z[cluster] -= wordFre * self.prob_on[documentID]
                (cluster, prob) = self.sampleCluster(document)
                if prob >= self.prob_add:
                    self.prob_on[documentID] = 1
                else:
                    self.prob_on[documentID] = 0
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += self.prob_on[documentID]
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre * self.prob_on[documentID]
                    self.n_z[cluster] += wordFre * self.prob_on[documentID]

    def predict_with_prob_raw(self, documents):
        self.assign_labels_with_prob_raw(documents)
        for i in range(self.iterNum):
            for document in documents:
                documentID = document.documentID
                cluster = self.z[documentID]
                self.ClusterNum = max(self.ClusterNum, cluster + 1)
                self.m_z[cluster] -= self.prob_on[documentID]
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre * self.prob_on[documentID]
                    self.n_z[cluster] -= wordFre * self.prob_on[documentID]
                (cluster, prob) = self.sampleCluster(document)
                self.prob_on[documentID] = prob
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += self.prob_on[documentID]
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre * self.prob_on[documentID]
                    self.n_z[cluster] += wordFre * self.prob_on[documentID]

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
        self.V_beta = self.V * self.beta
        prob = [float(0.0)] * self.ClusterNum
        for cluster in range(self.ClusterNum):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] + self.alpha
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
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.V_beta + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        kChoosed = -1
        k_max = -1
        for k in range(1, self.ClusterNum):
            if k_max < prob[k]:
                kChoosed = k
                k_max = prob[k]
        if kChoosed == -1:
            print("not choosed")
        return (kChoosed, k_max)