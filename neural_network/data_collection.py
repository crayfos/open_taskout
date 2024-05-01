import psycopg2
import re
from collections import defaultdict

from web.db_config import db_params

import string
from razdel import tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import torch
from sklearn.preprocessing import LabelEncoder

import pandas as pd

# Функция для очистки текста от HTML-тегов
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# SQL-запрос для получения данных
query = """
WITH numbered_rows AS (
    SELECT t.title, t.description, c.category_id,
           ROW_NUMBER() OVER(PARTITION BY c.category_name ORDER BY random()) AS row_num
    FROM tasks AS t
    JOIN task_processing AS tp ON t.task_id = tp.task_id
    JOIN categories AS c ON tp.category = c.category_id
    WHERE tp.status = (SELECT status_id FROM statuses WHERE status_name = 'manual')
	ORDER BY random()
)
SELECT title, description, category_id
FROM numbered_rows
WHERE row_num <= 500
ORDER BY category_id;
"""

# Выполняем запрос и обрабатываем данные
cur.execute(query)
data = defaultdict(list)

for title, description, category in cur.fetchall():
    # Очищаем описание от HTML-тегов
    description = clean_html(description) if description else ''
    # Объединяем заголовок и описание
    text = f"{title} {description}".strip()
    # Добавляем в словарь по категориям
    data[category].append(text)

# Закрываем соединение с базой данных
cur.close()
conn.close()

print(data.keys())
arr = data[1]

# Токенизация
tokens = list(tokenize(arr[0]))
tokens = [word.text for word in tokens]
print([_ for _ in tokens])

# Удаление знаков препинания
tokens = [word for word in tokens if word not in string.punctuation]
print(tokens)

# Удаление стоп-слов
stop_words = set(stopwords.words('russian'))
tokens = [word for word in tokens if word not in stop_words]
print(tokens)

# Стемминг
stemmer = SnowballStemmer("russian")
tokens = [stemmer.stem(word) for word in tokens]
print(tokens)



# Обработка для TextDataset
categories = list(data.keys())
label_encoder = LabelEncoder()
label_encoder.fit(categories)

# Создание списков текстов и меток
texts = []
labels = []

for category, text_list in data.items():
    for text in text_list:
        texts.append(text)
        labels.append(label_encoder.transform([category])[0])

# Преобразование меток в one-hot encoding
num_classes = len(categories)
labels_one_hot = torch.zeros((len(labels), num_classes))
labels_one_hot[range(len(labels)), labels] = 1

print(texts[0])

# Создание DataFrame
df = pd.DataFrame({'Texts': texts, 'Labels': labels})

# Сохранение DataFrame в CSV
df.to_csv('data.csv', index=False)

