import pandas as pd
import itertools
import string
import json

# reading the data , aka FinalResults collection which the result of exporting it to a json file

df = pd.read_json('D:\\visual_genome.json', lines=True)

data = list()
reg = list()
sent = ""
rellen = 0

for region in df['regions'] :
    for relationship in region['relationships'] :
        rellen = len(relationship)
        if len(relationship) > 0 :
            try : 
                sent = sent + str.lower(relationship['relationships']['relationships']['subject']['name']) + " "
            except :
                for name in relationship['relationships']['relationships']['subject']['names']:
                    sent = sent + str.lower(name) + " "
            
            sent = sent + str.lower(relationship['relationships']['relationships']['predicate']) + " "
            
            try : 
                sent = sent + str.lower(relationship['relationships']['relationships']['object']['name']) + " "
            except :
                for name in relationship['relationships']['relationships']['object']['names']:
                    sent = sent + str.lower(name) + " "
            
            reg.append(sent)
            sent = ""
            
    if rellen > 0 :
        reg.append((str.lower(region['phrase'])))
        data.append(reg)
        rellen = 0
    reg = list()
    
#delete duplicate instances , becuase they don't add any new info

data = list(k for k,_ in itertools.groupby(data))

puncts = string.punctuation.replace('\'', '')

temp = list()
tempdata = list()
for item in data :
    for s in item :
        x = ''.join(c for c in s if c not in puncts)
        temp.append(x)
    tempdata.append(temp)
    temp = list()
    
data = tempdata

with open('data.json', 'w') as f:
    json.dump(data, f)