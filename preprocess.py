

def proprecess_for_text(dataset):
    read_name = './dataset/' + dataset + '.txt'
    write_name = './dataset/' + dataset + '_.txt'
    stopwords = './stopwords/stopwords.txt'
    file_in = open(read_name,'r')
    file_out = open(write_name,'w')
    file_word = open(stopwords,'r')
    wordlist = (file_word.readline()).strip().split(' ')
    in_lines = file_in.readlines()
    for line in in_lines:
        out_word = []
        words = line.strip().split(' ')
        for word in words:
            if word not in wordlist:
                out_word.append(word)
        file_out.write(' '.join(out_word))

dataset = ['Biomedical','SearchSnippets','StackOverflow']

if __name__ == '__main__':
    for i in range(3):
        proprecess_for_text(dataset[i])