import psycopg2
import re
from collections import defaultdict
from web.db_config import db_params
import string
from razdel import tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


conn = psycopg2.connect(**db_params)
cur = conn.cursor()

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
WHERE row_num <= 1000
ORDER BY category_id;
"""

cur.execute(query)
data = defaultdict(list)

for title, description, category in cur.fetchall():
    description = clean_html(description) if description else ''
    text = f"{title} {description}".strip()
    data[category].append(text)

cur.close()
conn.close()

print(data.keys())
arr = data[1]


# ТЕСТИРОВАНИЕ ПРЕДОБРАБОТКИ

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




train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1)

# Сортировка
train_texts, train_labels = zip(*sorted(zip(train_texts, train_labels), key=lambda x: x[1]))
test_texts, test_labels = zip(*sorted(zip(test_texts, test_labels), key=lambda x: x[1]))

# Сохранение
df = pd.DataFrame({'Texts': train_texts, 'Labels': train_labels})
df.to_csv('train_data.csv', index=False)

df = pd.DataFrame({'Texts': test_texts, 'Labels': test_labels})
df.to_csv('test_data.csv', index=False)
