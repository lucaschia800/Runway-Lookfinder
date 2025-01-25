import vogue
import json
import time
import random
import designer_scrape
from concurrent.futures import ThreadPoolExecutor, as_completed

list_designers_names = designer_scrape.get_designers()

vogue_all = []

def designer_scraper(designer):
    list_shows = []
    dict_designer = {
        'Designer' : designer,
        'Shows': list_shows}
    show_curr = vogue.designer_to_shows(designer)
    for show in show_curr:
        dict_show = {
            'Show Name' : show
        }
        urls_looks = vogue.designer_show_to_download_images(designer, show)
        dict_show['Looks'] = urls_looks
        list_shows.append(dict_show)
    time.sleep(random.uniform(1, 2))
    print(designer)
    return dict_designer



with ThreadPoolExecutor(max_workers = 5) as executor:
    futures = [executor.submit(designer_scraper, designer) for designer in list_designers_names]

    #results = list(executor.map(scrape_designer, list_designers_names)) 
    #use this if I want to maintain designer order name in alphabetical order

    for future in as_completed(futures):
        result = future.result()
        if result:
            vogue_all.append(result)

with open('all_photos.json', 'w') as json_file:
    json.dump(vogue_all, json_file, indent=4) 
    


    
        

