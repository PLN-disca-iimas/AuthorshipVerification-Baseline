import json

def getDataJSON(route):
    with open(route,"r",encoding="utf-8") as f:
        result = [json.loads(jline.replace("same","value")) for jline in f.read().splitlines()]
    return result