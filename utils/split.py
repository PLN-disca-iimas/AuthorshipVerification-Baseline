import json

def getDataJSON(route):
    print('route: ', route)
    #with open(route,"r",encoding="utf-8") as f:
    #    result = [json.loads(jline.replace("same","value")) for jline in f.read().splitlines()]
    #return result
    result = []
    for line in open(route, encoding='utf8'):
        d = json.loads(line.strip().replace("same","value"))
        result.append(d)
    return result