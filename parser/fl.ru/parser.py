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
from web.parser.db_config import db_params
from web.parser.habr.habr_categories import categories_info

tasks_counter = 0

locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}


def url_exists_in_db(url, cursor):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM tasks WHERE url=%s)', (url,))
    return cursor.fetchone()[0]


def get_task_links(url):
    task_links = []
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        project_items = soup.find_all('div', class_='b-post__grid')

        for item in project_items:
            script_text = item.find_all('script')[2]
            if re.search(r'Заказ', script_text.string):
                title_element = item.find('h2', class_='b-post__title')
                if title_element and title_element.find('a'):
                    link = 'https://www.fl.ru' + title_element.find('a')['href']
                    if not url_exists_in_db(link, cursor):
                        task_links.append(link)
        return task_links
    except (Timeout, ConnectionError) as e:
        print(f"Возникла ошибка при подключении к {url}: {e}")
    except RequestException as e:
        print(f"Произошла ошибка при запросе к {url}: {e}")
    finally:
        cursor.close()
        conn.close()
    return ['error']


def get_published_date(meta_string):
    # Извлечь дату публикации из строки
    date_match = re.search(r'\d{2}\.\d{2}\.\d{4} \| \d{2}:\d{2}', meta_string)
    if date_match:
        published_date_str = date_match.group(0)
        published_date = datetime.strptime(published_date_str, '%d.%m.%Y | %H:%M')
        return published_date.strftime('%Y-%m-%d %H:%M:%S')
    return None


def parse_price(price_str):
    price_type_id = 1
    if 'заказ' in price_str:
        price_type_id = 3
    elif 'час' in price_str:
        price_type_id = 2

    range_type, max_price = 0, 0
    match = re.search(r"Бюджет:\s*(До|Более)?\s*(\d+(?:\s*\d+)*)\s*(?:[-—]\s*(\d+(?:\s*\d+)*))?", price_str)
    if match:
        first_price = match.group(2).replace(" ", "")
        second_price = match.group(3).replace(" ", "") if match.group(3) else None
        max_price = max(int(first_price), int(second_price)) if second_price else int(first_price)
        range_group = match.group(1)

        if range_group == "Более":
            range_type = 2
        elif range_group == "До" or second_price:
            range_type = 1

    return max_price, range_type, price_type_id


def get_task_details(task_url):
    try:
        response = requests.get(task_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', {'id': True}).get_text(strip=True)
        description = str(soup.find('div', {'id': True, 'class': 'text-5'}))
        description = re.sub(r'<(?!br\b|br\/\b)[^>]+>', '', description)
        description = re.sub(r'\xa0|&nbsp;|^\n*\s+|\s+\n*$', '', description)
        description = re.sub(r'(<br\/?>\s*){3,}', '<br><br><br>', description)

        price_div = soup.find('div', {'class': 'text-4'})
        price_str = price_div.get_text(strip=True)
        price, price_range_type, price_type_id = parse_price(price_str)

        meta_div = soup.find('div', {'class': 'b-layout__txt b-layout__txt_padbot_30 mt-32'})
        meta_str = meta_div.find('div', {'class': 'text-5'}).get_text(strip=True)
        published_date = get_published_date(meta_str)

        categories = [a.get_text(strip=True) for a in soup.find_all('a', {'data-id': 'category-spec'})]
        if not categories:
            standard_category, standard_subcategory = 'Прочее', 'Прочее'
        else:
            standard_category, standard_subcategory = categories[0], categories[1]

        task_details = {
            'url': task_url,
            'title': title,
            'description': description,
            'price': price,
            'price_range_type': price_range_type,
            'price_type_id': price_type_id,
            'published_date': published_date,
            'standard_category': standard_category,
            'standard_subcategory': standard_subcategory,
        }
        empty_fields = [key for key, value in task_details.items() if
                        not value and (key != 'price_range_type' and key != 'price')]
        if empty_fields:
            raise ValueError(f"Следующие поля пустые: {', '.join(empty_fields)}")
        return task_details
    except ValueError as ve:
        print(f"Получены неполные данные для задачи {task_url}: {ve}")


task_queue = queue.Queue()
print_lock = threading.Lock()
producer_finished_event = threading.Event()


def producer(retries=5):
    depth = 0
    page_number = 1
    url = 'https://www.fl.ru/projects/'

    while depth < 3:
        if retries == 0:
            print(f"Превышено максимальное количество попыток для {url}")
            break
        links = get_task_links(f"{url}page-{page_number}/")
        if links and links[0] != 'error':
            for link in links:
                task_queue.put((link, 0))
            page_number += 1
        elif links and links[0] == 'error':
            retries -= 1
            print(f"Повторная попытка {retries} для URL {url}")
        else:
            depth += 1
            print(f"Список новых заданий получен")
    producer_finished_event.set()


def save_task_to_db(task_details):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    task_id = None  # Инициализация переменной task_id
    try:
        insert_query = '''
        INSERT INTO tasks (url, title, description, price, price_type_id, published_date, standard_category, standard_subcategory)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING task_id;
        '''
        cursor.execute(insert_query, (
            task_details['url'],
            task_details['title'],
            task_details['description'],
            task_details['price'],
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
def save_task_processing(task_id, deep_category):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    try:
        category_id = deep_category
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
def consumer(retry_limit=3, delay_between_retries=5):
    while True:
        try:
            link, retries = task_queue.get(timeout=10)

            # Попытаемся получить детали задания
            task_info = get_task_details(link)

            with print_lock:
                print(task_info)
            # new_task = save_task_to_db(task_info)
            # if new_task is not None:
            #     save_task_processing(new_task, deep_category)

            global tasks_counter
            tasks_counter += 1
            if retries > 0:
                with print_lock:
                    print(f"Ссылка {link} обработана с попытки {retries}")

        except queue.Empty:
            time.sleep(10)
            producer_finished_event.wait()
            if task_queue.empty():
                return

        except Exception as e:
            with print_lock:
                print(f"Ошибка при обработке ссылки {link}: {e}, попытка {retries}")
            retries += 1
            if retries <= retry_limit:
                # Планируем повторную попытку с задержкой
                time.sleep(delay_between_retries)
                task_queue.put((link, retries))
            else:
                with print_lock:
                    print(f"Достигнут лимит попыток для ссылки {link}")
        finally:
            task_queue.task_done()


def start_habr_parser():
    global tasks_counter
    producer_finished_event.clear()
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Запускаем рабочие потоки потребителей
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(consumer) for _ in range(10)]

    # Дождемся завершения всех потребителей
    concurrent.futures.wait(futures)

    # Выводим итоговую информацию
    print(f"Обработано заданий: {tasks_counter}")
    tasks_counter = 0


if __name__ == "__main__":
    start_habr_parser()
