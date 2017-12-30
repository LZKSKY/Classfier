# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:47:27 2015

@author: jianhua
"""

#import simplejson as json
import json
import re
import codecs
from nltk.stem.wordnet import WordNetLemmatizer
import logging
import chardet
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)(x?)(\w+);', re.UNICODE)
CleanRule = r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt | rt |\n'
        
stopwordFile = './stopwords/lang_stopwords.txt'
inFile = 'newtweets.txt'
outFile = 'newtweets_new.txt'
outFileLowDP = 'newtweets_new_lowDP.txt'
#将仅出现在一个文档中的单词删除吗？这样处理是否合理？

def tokenize(doc):
    '''
    tokens = [token.encode('utf8') for token in tokenizeIter(doc, errors='ignore')
                            if 2 <= len(token) <= 15 and not token.startswith('_')]
    '''
    tokens = [token for token in tokenizeIter(doc, errors='ignore')
              if 2 <= len(token) <= 15 and not token.startswith('_')]
    return tokens

def tokenizeIter(text, errors="strict"):
    if not isinstance(text, str):
        text = str(text, encoding='utf8', errors=errors)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()

with open(stopwordFile, 'r') as fin:
    line = fin.readline()
    line = line.strip()
    stopwords = line.split(' ')

file_temp = open(inFile, 'r')
fin_length = float(len(file_temp.readlines()))
file_temp.close()

Stemmer = WordNetLemmatizer()
fout = open(outFile, 'w')
with open(inFile, 'r') as fin:
    schedule_percentage = float(0)
    for docJson in fin:
        schedule_percentage += 1
        print(round(schedule_percentage/fin_length, 4), "%", end="\r")
        #print docJson
        #docObj = json.loads(docJson.replace('\r\n', ''))
        docObj = json.loads(docJson)
        text = ' '.join(re.sub(CleanRule, " ", docObj['text']).split())
        '''
        words = [Stemmer.lemmatize(str(word)) for word in text.split(' ')
                                if word not in stopwords and len(word) > 1]
        '''
        words = [Stemmer.lemmatize(word) for word in list(tokenize(text))
                 if word not in stopwords and len(word) > 1]
        #'''
        wordsNostop = [word.lower() for word in words if word.lower() not in stopwords]
        docObj['textCleaned'] = ' '.join(wordsNostop)
        fout.write(json.dumps(docObj) + '\n')
fout.close()
print()
#remove words that DF=1
wordNum = 0
docNum = 0
wordDocOccurNumDict = {}
wordOccurNumDict = {}
with open(outFile, 'r') as fin:
    schedule_percentage = float(0)
    for docJson in fin:
        schedule_percentage += 1
        print(round(schedule_percentage/fin_length, 4), "%", end="\r")
        docObj = json.loads(docJson)
        docStr = docObj['textCleaned']
        wordList = docStr.split(' ')
        wordNum += len(wordList)
        docNum += 1
        wordOccuredList = []
        for word in wordList:
            if word not in wordOccuredList:
                wordOccuredList.append(word)
                if word in wordDocOccurNumDict:
                    wordDocOccurNumDict[word] += 1
                else:
                    wordDocOccurNumDict[word] = 1

            if word in wordOccurNumDict:
                wordOccurNumDict[word] += 1
            else:
                wordOccurNumDict[word] = 1
print()
fout = open(outFileLowDP, 'w')
with open(outFile, 'r') as fin:
    schedule_percentage = float(0)
    for docJson in fin:
        schedule_percentage += 1
        print(round(schedule_percentage/fin_length, 4), "%", end="\r")
        docObj = json.loads(docJson)
        words = docObj['textCleaned'].split(' ')
        wordsHighDF = []
        for word in words:
            try:
                if wordDocOccurNumDict[word] > 1:
                    wordsHighDF.append(word)
            except:
                print(word)
    #    wordsHighDF = [word for word in words if wordDocOccurNumDict[word] > 1]
        docObj['textCleaned'] = ' '.join(wordsHighDF)
        fout.write(json.dumps(docObj) + '\n') 
fout.close()

print('Done')