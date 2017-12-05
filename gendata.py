import json
import numpy as np

file_read = open('./data/newtweets','r')
cluster = {}
maxID = 0
for line in file_read:
    obj = json.loads(line)
    clusterID = int(obj['clusterNo'])
    if clusterID not in cluster:
        cluster[clusterID] = 1
    else:
        cluster[clusterID] += 1
    maxID = max(maxID, clusterID)
file_read.close()

cluster_array = []
for i in range(maxID + 1):
    if i in cluster:
        cluster_array.append([i, cluster[i]])

sorted_array = sorted(cluster_array, key = lambda a:a[1] , reverse= True)
top_ten = sorted_array[0:10]
top_ten_cluster = [i[0] for i in top_ten]
print(top_ten)
train_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ratio_len = len(train_ratio)
train_num = np.zeros([ratio_len, 10])
test_num = np.zeros([ratio_len, 10])
for i in range(ratio_len):
    train_num[i] = [train_ratio[i]*top_ten[j][1] for j in range(10)]
    test_num[i] = [test_ratio[i]*top_ten[j][1] for j in range(10)]

file_read = open('./data/newtweets','r')
file_write = open('./data/topten','w')
file_train = {}
file_test = {}
for i in range(ratio_len):
    file_train[i] = open('./data1/train' + str(train_ratio[i]), 'w')
    file_test[i] = open('./data1/test' + str(train_ratio[i]), 'w')
for line in file_read:
    obj = json.loads(line)
    clusterID = int(obj['clusterNo'])
    if clusterID in top_ten_cluster:
        tmp = -1
        for i in range(len(top_ten_cluster)):
            if top_ten[i][0] == clusterID:
                tmp = i
                break
        if tmp == -1:
            print('error in clusterid ' + str(clusterID))
            exit(0)
        obj['clusterNo'] = tmp
        new_obj = json.dumps(obj)
        file_write.write(new_obj + '\n')
        for i in range(ratio_len):
            if train_num[i][tmp] > 0:
                train_num[i][tmp] -= 1
                (file_train[i]).write(new_obj + '\n')
            if test_num[i][tmp] > 0:
                test_num[i][tmp] -= 1
            else:
                (file_test[i]).write(new_obj + '\n')

for i in range(len(train_ratio)):
    file_train[i].close()
    file_test[i].close()
file_read.close()
file_write.close()
