import json

with open('config.json', 'r') as f:
    pdict = json.load(fp = f)
    for client in pdict['Size of samples for labels in clients']:
        print(client)