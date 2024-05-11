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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import json

from fp.fp import FreeProxy

locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

def get_proxy():
    try:
        proxy = FreeProxy(timeout=1, https=True, rand=True).get()
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

proxy = get_proxy()
def get_task_links(url):
    task_links = []
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    while True:
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")

            chrome_options.add_argument(f'--proxy-server={proxy}')

            # Инициализация драйвера Selenium
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            driver.get(url)
            # Ожидание загрузки элементов на странице
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'wants-card__header-title'))
            )

            # Получение данных с помощью Selenium
            tasks = driver.find_elements(By.CLASS_NAME, 'wants-card__header-title')
            for task in tasks:
                link = task.find_element(By.TAG_NAME, 'a').get_attribute('href')
                if not url_exists_in_db(link, cursor):
                    task_links.append(link)
            driver.quit()
            break
        except TimeoutException as e:
            print(f"Превышено время ожидания элементов на странице {url}: {e}")
            driver.quit()
            global proxy
            proxy = get_proxy()  # Получаем новый прокси и повторяем попытку
        except (ConnectionError, RequestException) as e:
            print(f"Возникла ошибка при подключении к {url}: {e}")
            driver.quit()
            break
        finally:
            cursor.close()
            conn.close()
    return task_links if task_links else ['error']


def get_script_json_data(html_content):
    pattern = re.compile(r'window\.stateData\s*=\s*(\{.*?\});</script>', re.DOTALL)
    match = pattern.search(html_content)
    if match:
        json_data = match.group(1)
        json_data = json_data.replace('</span>', '')
        data = json.loads(json_data)
        return data
    return None

def get_task_details(task_url):
    response = requests.get(task_url, headers=headers, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Используем функцию для извлечения данных из скрипта
    script_data = get_script_json_data(response.text)
    if script_data:
        title = script_data['wantData']['name']
        description = script_data['wantData']['wantClearedDescription']

        price = float(script_data['wantData']['price_limit'])
        max_price = float(script_data['wantData']['wantGetPossiblePriceLimit'])
        price_range_type = 1
        if price != max_price:
            price_range_type = 2
            price = max_price
        price_type_id = 3

        published_date = datetime.strptime(script_data['wantData']['date_create'], '%Y-%m-%d %H:%M:%S')
        published_date = published_date.strftime('%Y-%m-%d %H:%M:%S')

        task_details = {
            'url': task_url,
            'title': title,
            'description': description,
            'price': price,
            'price_range_type': price_range_type,
            'price_type_id': price_type_id,
            'published_date': published_date,
            'standard_category': '',
            'standard_subcategory': ''
        }

        return task_details
    return None


task_queue = queue.Queue()
print_lock = threading.Lock()
producer_finished_event = threading.Event()


def producer(categories_info, max_retries=5):
    retries = {url: 0 for deep_category, category, subcategory, url in categories_info}
    page_numbers = {url: 1 for deep_category, category, subcategory, url in categories_info}
    categories_info_copy = categories_info.copy()

    while categories_info_copy:
        for deep_category, category, subcategory, url in list(categories_info_copy):
            page_num = page_numbers[url]
            if retries[url] < max_retries:
                links = get_task_links(f"{url}&page={page_num}")
                if links and links[0] != 'error':
                    for link in links:
                        task_queue.put((link, deep_category, category, subcategory, 0))
                    page_numbers[url] += 1
                elif links and links[0] == 'error':
                    retries[url] += 1
                    print(f"Повторная попытка {retries[url]} для URL {url}")
                else:
                    categories_info_copy.remove((deep_category, category, subcategory, url))
            else:
                print(f"Превышено максимальное количество попыток для {url}")
                categories_info_copy.remove((deep_category, category, subcategory, url))
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
            link, deep_category, category, subcategory, retries = task_queue.get(timeout=10)

            # Попытаемся получить детали задания
            task_info = get_task_details(link)
            task_info['standard_category'] = category
            task_info['standard_subcategory'] = subcategory

            with print_lock:
                print(task_info)
            # new_task = save_task_to_db(task_info)
            # if new_task is not None:
            #     save_task_processing(new_task, deep_category)
            tasks_data.append(task_info)
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
                print(f"Ошибка при обработке ссылки {link}: {e}, попытка {retries}")
            retries += 1
            if retries <= retry_limit:
                # Планируем повторную попытку с задержкой
                time.sleep(delay_between_retries)
                task_queue.put((link, deep_category, category, subcategory, retries))
            else:
                with print_lock:
                    print(f"Достигнут лимит попыток для ссылки {link}")


tasks_data = []


def start_habr_parser():
    producer_finished_event.clear()
    producer_thread = threading.Thread(target=producer, args=(categories_info,))
    producer_thread.start()

    # Запускаем рабочие потоки потребителей
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(consumer) for _ in range(10)]

    # Дождемся завершения всех потребителей
    concurrent.futures.wait(futures)

    # Выводим итоговую информацию
    print(f"Обработано заданий: {len(tasks_data)}")
    tasks_data.clear()


if __name__ == "__main__":
    start_habr_parser()
