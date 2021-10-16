import jsonlines
import json
import os

ruta_ = 'test-corpus/test/'
lst_arch = os.listdir(ruta_)
lst_arch = [x for x in lst_arch if x[:2]=='EN']
dic = []
ruta_salida = 'pairs.json'
for i in range(len(lst_arch)):
    f1 = open (ruta_ + '%s/known01.txt'%lst_arch[i],'r')
    f2 = open (ruta_ + '%s/unknown.txt'%lst_arch[i],'r')
    text1 = f1.read()
    text2 = f2.read()
    dict_json = {"id": "%s"%lst_arch[i], "pair": [text1, text2]}
    dic.append(dict_json)
    f2.close()
    f1.close()

with open(ruta_salida, mode='w') as writer:
    json.dump(dic, writer, ensure_ascii=False)
   

with open(ruta_salida, mode='r') as r:
    d = json.load(r)
    i=0
    for x in d:
        text = []
        for item in x['pair']:
            item = item.replace('\n','').replace('\"','"').replace('\'','').replace('\ufeff','')
            text.append(item)
        d[i]['pair'] = text
        i+=1

with jsonlines.open(ruta_ + 'pairs.jsonl', mode='w') as writer:
    for i in range(len(lst_arch)):
        writer.write(d[i])