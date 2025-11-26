import json
from mimesis import Person, Gender as mimesis_Gender
from mimesis.locales import Locale
import re
import requests
import os
import time
from pytrovich.enums import NamePart, Gender, Case
from pytrovich.maker import PetrovichDeclinationMaker
from pytrovich.detector import PetrovichGenderDetector



API_URL = "https://house.fandom.com/ru/api.php"

def fetch_article(title):
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        data = response.json()
        if "error" in data:
            print(f"Ошибка при загрузке '{title}': {data['error']['info']}")
            return None
        html = data["parse"]["text"]["*"]
        return html
    except Exception as e:
        print(f"Ошибка при загрузке '{title}': {e}")
        return None

def fetch_arcticle_list(cat_name, articles, limit):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": cat_name,
        #"cmtitle": "Категория:Работники Принстон-Плейнсборо",
        "format": "json",
        "cmlimit": 30
    }

    count= 0
    
    while len(articles) < limit:
        try:
            response = requests.get(API_URL, params=params, timeout=10)
            data = response.json()
            if "error" in data:
                print(f"Ошибка при загрузке '{cat_name}': {data['error']['info']}")
                return None
            #print(json.dumps(data, indent=4, ensure_ascii=False))
            for per in data['query']['categorymembers']:
                if per['title'][:9] == 'Категория':
                    fetch_arcticle_list(per['title'], articles, limit)
                elif per['title'] in articles:
                    f = 1
                else:
                    articles.append(per['title'])
        except Exception as e:
            print(f"Ошибка при загрузке '{cat_name}': {e}")
            return None

        params['cmcontinue'] = data['continue']['cmcontinue']

def html_to_text(html):
    import re
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\[\d+\]|\[править\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_articles(path):
    articles = []
    fetch_arcticle_list("Категория:Персонажи", articles, 40)
    fetch_arcticle_list("Категория:Эпизоды", articles, 40)

    os.makedirs(path, exist_ok=True)

    for title in articles:
        html = fetch_article(title)
        if html:
            safe_filename = title.replace("\\", "_").replace('?', '_').replace("/", "_")
            file_name = f"{path}/{safe_filename}.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(html_to_text(html))
        
            time.sleep(0.6)

def get_persons():
    articles = []
    fetch_arcticle_list("Категория:Персонажи", articles, 400)
    fetch_arcticle_list("Категория:Актёрский_состав", articles, 400)

    return articles

def get_name_parts(name):
    name_parts = name.split(" ")
    n = len(name_parts)
    
    fn  = name_parts[0]
    ln    = name_parts[-1] if n >= 2 else None
    mn = name_parts[1] if n == 3 else None

    return fn, mn, ln, n

def map_persons(persons):
    person_mapping = {}
    fake_person = Person(Locale.RU)
    
    #choices={
    #    mimesis_Gender.MALE: 0.2,
    #    mimesis_Gender.FEMALE: 0.8,
    #}
    detector = PetrovichGenderDetector()

    for person in persons:
        first_name, middle_name, last_name, n = get_name_parts(person)
        try:
            gender_ = mimesis_Gender[detector.detect(firstname=first_name, lastname=last_name, middlename=middle_name).name]
        except Exception as e:
            gender_ = mimesis_Gender.MALE

        #person_mapping[person] = fake_person.full_name(gender=fake_person.random.weighted_choice(choices=choices))
        person_mapping[person] = fake_person.full_name(gender=gender_)
    
    return person_mapping

def name_declensions(name):
    
    first_name, middle_name, last_name, n = get_name_parts(name)

    detector = PetrovichGenderDetector()
    try:
        gender_ = detector.detect(firstname=first_name, lastname=last_name, middlename=middle_name)
    except Exception as e:
        gender_ = Gender.MALE

    
    maker = PetrovichDeclinationMaker()
    declensions = [{'parts_count':n, 'name':name, 'fn':first_name, 'ln':last_name, 'mn':middle_name}]
    for case in Case:
        fn = maker.make(NamePart.FIRSTNAME, gender_, case, first_name)
        ln = maker.make(NamePart.LASTNAME, gender_, case, last_name) if n > 1 else ''  
        mn = maker.make(NamePart.MIDDLENAME, gender_, case, middle_name) if n == 3 else ''
        declensions.append({'parts_count':n, 'name':' '.join(filter(None, [fn, mn, ln])),'fn':fn, 'ln':ln, 'mn':mn})

    return declensions
    

def extend_mapping(person_mapping:dict):
    
    mapping = {}

    for old, new in person_mapping.items():
        old_names = name_declensions(old)
        new_names = name_declensions(new)
        
        for i in range(len(old_names)):
            mapping[old_names[i]['name']] = new_names[i]['name']
            if old_names[i]['parts_count'] > 1 and new_names[i]['parts_count'] > 1:
                if len(old_names[i]['ln']) > 3:
                    mapping[old_names[i]['ln']] = new_names[i]['ln']
                    mapping[old_names[i]['fn']] = new_names[i]['fn']

    return mapping
   

def aplly_terms_map_to_text(text:str, terms_mapping:dict):

    mapped_text = text

    for old in sorted(terms_mapping.keys(), key=len, reverse=True):
        pattern = rf'\b{re.escape(old)}\b'
        mapped_text = re.sub(pattern, terms_mapping[old], mapped_text, flags=re.IGNORECASE)

    return mapped_text

   

def aplly_terms_map(articles_path, terms_mapping, result_folder):

    os.makedirs(result_folder, exist_ok=True)

    for filename in os.listdir(articles_path):
        if filename.endswith(".txt"):
            with open(os.path.join(articles_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
            
            new_text = aplly_terms_map_to_text(text, terms_mapping)

            with open(os.path.join(result_folder, filename), 'w', encoding='utf-8') as file:
                file.write(new_text)

            os.remove(os.path.join(articles_path, filename))

           

def main():
    articles_path = './knowledge_base/raw' 
    #fetch_articles(articles_path)

    persons = get_persons()
    persons_mapping = map_persons(persons)

    terms_map = extend_mapping(persons_mapping)
    terms_map['Принстон Плейнсборо'] = 'Областной Мытищинский'
    terms_map['Принстон-Плейнсборо'] = 'Областной Мытищинский'
    terms_map['Плейнсборо'] = 'Областной Мытищинский'
    terms_map['Принстон'] = 'Областной Мытищинский'

    with open('./knowledge_base/terms_map.json', "w", encoding="utf-8") as file:
        json.dump(terms_map, file, ensure_ascii=False, indent=4)

    aplly_terms_map(articles_path, terms_map, './knowledge_base')

    os.removedirs(articles_path)

if __name__ == "__main__":
    main()