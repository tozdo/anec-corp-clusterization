import urllib.request
import re
from urllib.parse import quote
import html
import os

pers = ['Абрамович', 'Адам и Ева', 'Баба-Яга', 'Березовский', \
        'Билл Гейтс', 'Брежнев', 'Буратино', 'Валуев', 'Винни-Пух', 'вовочка',\
        'Гарри Поттер', 'Дед Мороз', 'Донцова', 'Иван-царевич', 'Каренина',\
        'Карлсон', 'Колобок', 'Красная Шапочка', 'Куклачев', 'Мазай',  \
        'Медведев', 'Навальный', 'Обама', 'Онищенко', 'Перельман', 'Прохоров', \
        'Путин', 'Пушкин', 'Рабинович', 'Ржевский', 'Сталин', 'Сусанин', \
        'Трамп', 'Хоттабыч', 'Чак Норрис', 'чапаев', 'Чебурашка', \
        'чукча', 'Шерлок Холмс', 'Штирлиц']

site = 'https://www.anekdot.ru/tags/'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'

def a_corp(): #создание корпуса анекдотов
    anecs = [] 
    for per in pers:
        for i in range(1, 67): 
            try:
                ssil = site + quote(per) + '/' + str(i) 
                req = urllib.request.Request(ssil, headers={'User-Agent':user_agent})
                with urllib.request.urlopen(req) as response:
                    hetemel = response.read().decode('utf-8')
                    anecs_per = re.findall('<div class="text">(.+?)</div>', hetemel) #все анеки от персонажа
                    n = 1
                    for anec in anecs_per:
                        anec = html.unescape(anec)
                        regTag = re.compile('<.*?>', flags=re.U | re.DOTALL)
                        clean_anec = regTag.sub("", anec)
                        adr = '/home/tozdo/anecs/' + per
                        if not os.path.exists(adr):
                            os.makedirs(adr)
                        file_name = adr + '/' + per[:3] + '_' + str(i)+ '_' + str(n)
                        if not clean_anec in anecs: 
                            f = open(file_name, 'w')
                            f.write(clean_anec)
                            f.close()
                            anecs.append(clean_anec)
                        n += 1
            except:
                print('Error at ' + str(i))

a_corp()


