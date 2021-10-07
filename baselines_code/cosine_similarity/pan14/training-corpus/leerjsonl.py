import jsonlines
import json
with jsonlines.open('english-novels/truth.jsonl') as r:
    i=0
    for o in r:
        print(o)
        i+=1
        if i==2:
            break