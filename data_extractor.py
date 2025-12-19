import requests
from bs4 import BeautifulSoup
page = requests.get("https://www.gutenberg.org/files/1756/old/vanya10h.htm") #Insert the link where the play is located
soup = BeautifulSoup(page.content, 'html.parser')

#Find the ending range for each play
ronju = []
for i in range(18,643): #Inspect the webpage to locate which tag and the number of tags where the data is located
    ronju.append(soup.select('p')[i].text)

#Find if there are any tuples
tup = 0
st = 0
other = 0
for i in range(0,len(ronju)):
    if type(ronju[i]) == tuple:
        tup = tup+1
    elif type(ronju[i]) == str:
        st = st+1
    else:
        other=other+1
print(tup, st, other)

#Extracting the text from the site
res = []
for i in ronju:
    if '\n' in i:
        res.append(i.replace("\n", ' '))
    else:
        res.append(i)

#Removing digits if in any text
for i in range(0,len(res)):
    res[i] = '' .join((z for z in res[i] if not z.isdigit()))
    res[i] = res[i].replace('  ',' ')
    
#Creating the dataframe
import pandas as pd
import numpy as np
rnj = pd.DataFrame()
rnj['text'] = res

#Extraction of the speaker
speaker = []
modifiedtext = []
for i in range(0,rnj.shape[0]):
    spl = rnj['text'][i].split('.')
    who = spl[0]
    what = '.'.join(spl[1::]).lstrip()
    speaker.append(who)
    modifiedtext.append(what)

rnj['speaker'] = speaker
rnj['actualtext'] = modifiedtext

#Replacing words from the vocab to convert to modern English
import json
with open('vocab.txt') as f:
    data = f.read()
lookup_dict = json.loads(data)

import re
s = ""
modern = []
for i in range(0,rnj.shape[0]):
    s = rnj['actualtext'][i]
    s = s.lower()
    for key in lookup_dict:
        newkey = ' '+key
        if re.search(newkey, s) != None:
            s = s.replace(newkey, ' '+lookup_dict[key])
    modern.append(s)
rnj['modern english'] = modern

#Removing the unnecessary brackets
brakremoval = rnj['modern english'].tolist()
for i in range(0,len(brakremoval)):
    brakremoval[i] = " ".join(re.sub("\(.*?\)|\[.*?\]","",brakremoval[i]).split())
rnj['complete text'] = brakremoval
