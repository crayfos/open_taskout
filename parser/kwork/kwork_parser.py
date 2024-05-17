from bs4 import BeautifulSoup
from datetime import datetime
import re
import locale

import queue
import threading
import concurrent.futures
import time
import requests
from requests.exceptions import Timeout, ConnectionError, RequestException

import psycopg2
from web.db_config import db_params
from web.parser.kwork.kwork_categories import categories_info
from web.parser.kwork.kwork_categories import subcategories_info

import json

from fp.fp import FreeProxy

locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}


def get_proxy():
    try:
        proxy = FreeProxy(timeout=1, https=True).get()
        while not check_proxy(proxy):
            proxy = FreeProxy(timeout=1, https=True).get()
        print(f"Используем прокси {proxy}")
        return proxy
    except Exception as e:
        print(f"Не удалось получить прокси: {e}")
        return None


def check_proxy(proxy_url):
    try:
        response = requests.get('https://kwork.ru/projects', proxies={'http': proxy_url, 'https': proxy_url}, timeout=5)
        if response.status_code == 200 and response.content:
            return True
        else:
            return False
    except requests.exceptions.ProxyError:
        print(f'Прокси {proxy_url} не работает. Ошибка прокси.')
        return False
    except requests.exceptions.ConnectTimeout:
        print(f'Прокси {proxy_url} не работает. Превышено время ожидания.')
        return False
    except requests.exceptions.RequestException as e:
        print(f'Прокси {proxy_url} не работает. Ошибка: {e}')
        return False


def url_exists_in_db(url, cursor):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM tasks WHERE url=%s)', (url,))
    return cursor.fetchone()[0]


categories = {}


def get_tasks_json(url, proxy):
    task_json = []
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    while True:
        proxies = {
            'http': proxy[0],
            'https': proxy[0]
        }
        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=15)
            script_data = get_script_json_data(response.text)
            global categories
            if categories == {}:
                categories = script_data['categories']
            script_data = script_data['wantsListData']['wants']

            for task in script_data:
                link = 'https://kwork.ru/projects/' + str(task['id'])
                if not url_exists_in_db(link, cursor):
                    task_json.append(task)
            cursor.close()
            conn.close()
            break
        except requests.exceptions.ProxyError:
            print(f"Возникла ошибка при подключении к {url}")
            proxy[0] = get_proxy()
            continue
        except Exception as e:
            print(f"Возникла ошибка при подключении к {url}: {e}")
            continue
    return task_json if task_json else []


def get_script_json_data(html_content):
    pattern = re.compile(r'window\.stateData\s*=\s*(\{.*?\});</script>', re.DOTALL)
    match = pattern.search(html_content)
    if match:
        json_data = match.group(1)
        json_data = json_data.replace('</span>', '')
        data = json.loads(json_data)
        return data
    return None


def get_standard_categories(category_id):
    global categories
    standard_category = None
    standard_subcategory = None

    for cat_id, cat_info in categories.items():
        if 'cats' in cat_info:
            for subcat in cat_info['cats']:
                if subcat['CATID'] == category_id:
                    standard_category = cat_info['name']
                    standard_subcategory = subcat['name']
                    return standard_category, standard_subcategory
        if cat_id == category_id:
            standard_category = cat_info['name']
            break

    return standard_category, standard_subcategory


def get_task_details(script_data):
    title = script_data['name']
    title = re.sub(r'\[:.*?\]|&.*?;', '', title)
    description = script_data['description'].replace('\n', '<br/>')
    description = re.sub(r'\[:.*?\]|&.*?;', '', description)
    description = re.sub(r'(<br\/?>\s*){3,}', '<br><br><br>', description)

    price = float(script_data['priceLimit'])
    max_price = float(script_data['possiblePriceLimit'])
    price_range_type = 1
    if price != max_price:
        price_range_type = 2
        price = max_price
    price_type_id = 3

    published_date = datetime.strptime(script_data['date_create'], '%Y-%m-%d %H:%M:%S')
    published_date = published_date.strftime('%Y-%m-%d %H:%M:%S')

    category_id = script_data['category_id']
    standard_category, standard_subcategory = get_standard_categories(category_id)
    # standard_category = script_data['parentCategoryName']
    # standard_subcategory = script_data['categoryName']

    task_details = {
        'url': 'https://kwork.ru/projects/' + str(script_data['id']),
        'title': title,
        'description': description,
        'price': price,
        'price_range_type': price_range_type,
        'price_type_id': price_type_id,
        'published_date': published_date,
        'standard_category': standard_category,
        'standard_subcategory': standard_subcategory
    }

    return task_details


task_queue = queue.Queue()
print_lock = threading.Lock()
producer_finished_event = threading.Event()


def producer(retries=5):
    depth = 0
    page_number = 1
    url = 'https://kwork.ru/projects?c=all'
    proxy = [get_proxy()]

    while depth < 3:
        if retries == 0:
            break

        links = get_tasks_json(f"{url}&page={page_number}", proxy)
        if links and links[0] != 'error':
            for link in links:
                task_queue.put((link, 0))
            page_number += 1
        elif links and links[0] == 'error':
            retries -= 1
            print(f"Повторная попытка {retries} для URL {url}&page={page_number}")
        else:
            print(f"Новых заданий на URL {url}&page={page_number} больше нет")
            depth += 1
            page_number += 1
    producer_finished_event.set()


def save_task_to_db(task_details):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    task_id = None  # Инициализация переменной task_id
    try:
        insert_query = '''
        INSERT INTO tasks (url, title, description, price, price_range_type, price_type_id, published_date, standard_category, standard_subcategory)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING task_id;
        '''
        cursor.execute(insert_query, (
            task_details['url'],
            task_details['title'],
            task_details['description'],
            task_details['price'],
            task_details['price_range_type'],
            task_details['price_type_id'],
            task_details['published_date'],
            task_details['standard_category'],
            task_details['standard_subcategory']
        ))
        task_id = cursor.fetchone()[0]  # Получаем task_id возвращаемый из запроса
        conn.commit()
    except Exception as e:
        print(
            f"Ошибка при вставке в БД: {e}\n info: {task_details['url'], task_details['standard_category'], task_details['standard_subcategory']}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
    return task_id  # Возвращаем task_id


# Этот код предполагает, что функция save_task_to_db возвращает task_id новой задачи
def save_task_processing(task_id, task_info):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    try:
        category_id = 0
        for subcategory_info in subcategories_info:
            if (task_info['standard_category'] == subcategory_info[1] and
                    task_info['standard_subcategory'] == subcategory_info[2]):
                category_id = subcategory_info[0]
        if category_id == 0:
            for category_info in categories_info:
                if task_info['standard_category'] == category_info[1]:
                    category_id = category_info[0]
        if category_id == 0:
            category_id = 6

        status_id = 2
        insert_query = '''INSERT INTO task_processing (task_id, category, status, category_change_date)
                          VALUES (%s, %s, %s, %s)'''
        category_change_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(insert_query, (task_id, category_id, status_id, category_change_date))
        conn.commit()
    except Exception as e:
        print(f"Ошибка при вставке в БД: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


# Функция consumer извлекает данные из очереди и обрабатывает каждую задачу
def consumer(retry_limit=3):
    while True:
        try:
            link, retries = task_queue.get(timeout=10)

            # Попытаемся получить детали задания
            task_info = get_task_details(link)

            # with print_lock:
            #     print(task_info)
            new_task = save_task_to_db(task_info)
            if new_task is not None:
                save_task_processing(new_task, task_info)

            global tasks_counter
            tasks_counter += 1
            if retries > 0:
                with print_lock:
                    print(f"Ссылка {link} обработана с попытки {retries}")
            task_queue.task_done()
        except queue.Empty:
            time.sleep(10)
            if producer_finished_event.is_set():
                return
            else:
                continue

        except Exception as e:
            with print_lock:
                print(f"Ошибка {e} при обработке ссылки {link}, попытка {retries}")
            retries += 1
            if retries <= retry_limit:
                task_queue.put((link, retries))
            else:
                with print_lock:
                    print(f"Достигнут лимит попыток для ссылки {link}")


tasks_counter = 0


def start_kwork_parser():
    global tasks_counter
    producer_finished_event.clear()
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Запускаем рабочие потоки потребителей
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(consumer) for _ in range(2)]

    # Дождемся завершения всех потребителей
    concurrent.futures.wait(futures)

    # Выводим итоговую информацию
    print(f"Обработано заданий: {tasks_counter}")
    tasks_counter = 0


if __name__ == "__main__":
    start_kwork_parser()
