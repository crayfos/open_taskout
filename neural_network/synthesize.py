from g4f.client import Client
import psycopg2
from web.db_config import db_params
import re
import pandas as pd

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

from collections import Counter


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def getTask():
    df = pd.read_csv('data.csv')
    texts = df['Texts'].tolist()
    labels = df['Labels'].tolist()

    return labels, texts


def getAnswer(task):
    prompt = f'''
        Ты – рекрутер на площадке труда. Тут собираются задания, которые возможно выполнить удалённо. Твоя задача – подбирать категорию для задания таким образом, чтобы его увидело как можно больше потенциальных исполнителей в диджитал. Ознакомься с категориями, на которые ты делишь задания:
        1) Разработка. Это задания, которые относятся к программированию, разработке сайтов, игр, настройке 1С и Bitrix, тестирование ПО и прочее.
        2) Дизайн. Тут собраны задачи по созданию айдентики, брендбука, логотипа или иконок, полиграфия, обложки для маркетплейсов, баннеры, дизайн презентаций, дизайн интерфейсов и сайтов, разработка сайтов на no-code платформах вроде Тильды и так далее.
        3) Контент. Задания, связанные с придумыванием контента и редактированием. Ведение социальных сетей, копирайтинг, написание статей, обработка и фотошоп фото, работа с аудио и видео, иллюстрация, анимация и 3D моделирование. Так же тут есть простые действия вроде поиска информации, составления таблиц и звонков по базе
        4) Маркетинг. Здесь собраны задания по поисковой оптимизации, SEO, аналитика и SRM, размещение на рекламных площадках, исследования, маркетинговые стратегии, нейминг, продвижение на маркетплейсах и соцсетях и так далее
        5) Бизнес. Это управление проектами, бизнес-аналитика, написание ТЗ и документации, найм персонала, бизнес-процессы, финансы и бухгалтерия, юриспруденция
        6) Другое. Сюда попадают задания, которые не подходят ни одной из категорий. То, что нельзя выполнить удалённо, относится к инженерному производству, чертежам, ремонту, ландшафтный дизайн, дизайн квартир, кафе и так далее

        Теперь обработай первое задание, вот оно:
        "{task}"

        Напиши, к какой категории оно относится и почему. Отвечай в формате [номер категории]: [причина]
        '''

    client = Client()

    list = []
    for i in range(3):
        match = None
        while not match:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            match = re.search(r'\d', response.choices[0].message.content)
        gpt_category = int(match.group(0))
        list.append(gpt_category)
    return most_frequent(list)


counter = 0
for category, task in zip(*getTask()):
    gpt_category = getAnswer(task)
    if gpt_category != category + 1:
        print('[', counter, '] ', gpt_category, ': ', category + 1, task)
    else:
        print('[', counter, ']')
    counter += 1

