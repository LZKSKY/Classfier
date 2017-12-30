import json
def json2txt(file_name):
    file_in = open(file_name + '_new_new.txt', 'r')
    doc_list = file_in.readlines()
    file_in.close()
    text_out = open(file_name + '_text.txt','w')
    for line in doc_list:
        obj = json.loads(line.strip())
        text_out.write(obj["textCleaned"] + '\n')
    text_out.close()


def j2t():
    file_in = open('./data/newtweets','r')
    doc_list = file_in.readlines()
    file_in.close()
    text_out = open('./data/newtweets_text', 'w')
    for line in doc_list:
        obj = json.loads(line.strip())
        text_out.write(obj["textCleaned"] + '\n')
    text_out.close()


path = './dataset/'
file_name_list = ['Biomedical','SearchSnippets','StackOverflow']

if __name__ == '__main__':
    # for i in range(len(file_name_list)):
    #     json2txt(path + file_name_list[i])
    j2t()