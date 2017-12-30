import json
import numpy as np
import matplotlib.pyplot as plt
def read_from_json():
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
    '''
    for i in range(len(sorted_array)):
        print(sorted_array[i][1], end=' ')
        if (i - 1)%10 == 0:
            print('')
    plt.plot([i for i in range(len(sorted_array))],[i[1] for i in sorted_array],'r-')
    for i in range(len(sorted_array)):
        if sorted_array[i][1] > 100:
            i += 1
        else:
            break
    print(i)
    plt.plot([i],100 ,'g+')
    plt.show()
    exit(0)
    '''
    cluster100 = 0
    for i in range(len(sorted_array)):
        if sorted_array[i][1] > 99:
            cluster100 += 1
        else:
            break;

    top_cluster_tuple = sorted_array[0:cluster100]
    top_cluster = [i[0] for i in top_cluster_tuple]
    # print(top_cluster_tuple)
    train_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio_len = len(train_ratio)
    train_num = np.zeros([ratio_len, cluster100])
    test_num = np.zeros([ratio_len, cluster100])
    for i in range(ratio_len):
        train_num[i] = [train_ratio[i]*top_cluster_tuple[j][1] for j in range(cluster100)]
        test_num[i] = [test_ratio[i]*top_cluster_tuple[j][1] for j in range(cluster100)]

    file_read = open('./data/newtweets','r')
    file_write = open('./new_tweets/topten','w')
    file_train = {}
    file_test = {}
    for i in range(ratio_len):
        file_train[i] = open('./new_tweets/train' + str(train_ratio[i]), 'w')
        file_test[i] = open('./new_tweets/test' + str(train_ratio[i]), 'w')
    for line in file_read:
        obj = json.loads(line)
        clusterID = int(obj['clusterNo'])
        if clusterID in top_cluster:
            tmp = -1
            for i in range(cluster100):
                if top_cluster_tuple[i][0] == clusterID:
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

def read_stc(dataset):
    text_file = open('./dataset/'+ dataset + '_text.txt','r')
    label_file = open('./dataset/'+ dataset + '_gnd.txt','r')
    text_lines = text_file.readlines()
    label_lines = label_file.readlines()
    text_file.close()
    label_file.close()
    if len(text_lines) != len(label_lines):
        print('word and label file have not same length')
        exit(-1)
    doc_length = len(text_lines)
    cluster = {}
    maxID = 0
    for i in range(doc_length):
        clusterID = int(label_lines[i])
        if clusterID not in cluster:
            cluster[clusterID] = 1
        else:
            cluster[clusterID] += 1
        maxID = max(maxID, clusterID)

    cluster_array = []
    for i in range(maxID + 1):
        if i in cluster:
            cluster_array.append([i, cluster[i]])

    train_ratio = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio_len = len(train_ratio)
    train_num = np.zeros([ratio_len, maxID + 1])
    for i in range(ratio_len):
        train_num[i,1:] = [train_ratio[i]*cluster[j] for j in range(1, maxID + 1)]

    file_train_text = {}
    file_test_text = {}
    file_train_label = {}
    file_test_label = {}
    for i in range(ratio_len):
        file_train_text[i] = open('./dataset/'+ dataset + '/train_text' + str(train_ratio[i]) + '.txt', 'w')
        file_test_text[i] = open('./dataset/' + dataset + '/test_text' + str(train_ratio[i]) + '.txt', 'w')
        file_train_label[i] = open('./dataset/' + dataset + '/train_label' + str(train_ratio[i]) + '.txt', 'w')
        file_test_label[i] = open('./dataset/' + dataset + '/test_label' + str(train_ratio[i]) + '.txt', 'w')
    for i in range(doc_length):
        clusterID = int(label_lines[i])
        for j in range(ratio_len):
            if train_num[j][clusterID] > 0:
                train_num[j][clusterID] -= 1
                (file_train_text[j]).write(text_lines[i])
                (file_train_label[j]).write(label_lines[i])
            else:
                (file_test_text[j]).write(text_lines[i])
                (file_test_label[j]).write(label_lines[i])

    for i in range(len(train_ratio)):
        file_train_text[i].close()
        file_test_text[i].close()
        file_train_label[i].close()
        file_test_label[i].close()

dataset = ['Biomedical','SearchSnippets','StackOverflow']

if __name__ == '__main__':
    # read_from_json(dataset)
    for i in range(len(dataset)):
        read_stc(dataset[i])