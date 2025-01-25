import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from unidecode import unidecode
    
def get_designers():
    r = requests.get("https://www.vogue.com/fashion-shows/designers")

    soup = BeautifulSoup(r.content, 'html5lib') 

    js = str(soup.find_all('script', type='text/javascript')[3])
    js = js.split(' = ')[1]
    js = js.split(';<')[0]
    data = json.loads(js)

    
    t = data['transformed']
    d = t['allRunwayDesigners']['groupedLinks']
    list_designers = []
    for letter in d:
        for designer in letter['links']:
            list_designers.append(designer['text'])

    return list_designers





