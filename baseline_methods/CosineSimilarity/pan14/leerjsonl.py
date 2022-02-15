import jsonlines
import json
with jsonlines.open('test-corpus/test/truth.jsonl') as r:
    i=0
    for o in r:
        print(o)
        i+=1
        if i==2:
            break