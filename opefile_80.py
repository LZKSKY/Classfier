import os
import json

path = '../TweetsEvents'
files = os.listdir(path)
files = files[0:-1]
outpath = './dataset_80/'
text_out = outpath + 'data'
label_out = outpath + 'label'
text_file = open(text_out,'w')
label_file = open(label_out,'w')
counter = 0
print(files)
print(len(files))
exit(0)
for i in range(len(files)):
    if files[i][0] == '.':
        continue
    filename = path + '/' + files[i]
    try:
        file = open(filename,'r',encoding='utf-8')
    except:
        continue
    for line in file:
        obj = json.loads(line)
        if 'text' not in obj:
            continue
        try:
            text_file.write(obj['text'])
            label_file.write(str(counter))
        except:
            pass
    counter += 1
    file.close()
text_file.close()
label_file.close()