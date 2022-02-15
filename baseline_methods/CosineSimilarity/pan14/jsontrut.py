import jsonlines
import json

ruta = 'test-corpus/test/truth.json'
ruta_ = 'test-corpus/test/truth.jsonl'
dic = []
with open(ruta, mode='r') as r:
    d = json.load(r)
    i=0
    for x in d['problems']:
        print(x)
        for item in x["answer"]:
            if item == "Y":
                t = 1
            else:
                t = 0
        dict_ = {"id":x['name'],"value":t}
        dic.append(dict_)
        i+=1

with jsonlines.open(ruta_, mode='w') as writer:
    for i in range(len(dic)):
        writer.write(dic[i])