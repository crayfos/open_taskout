import re
import locale

import queue
import threading
import concurrent.futures
import time
from datetime import datetime, timedelta
import requests

from web.db_config import db_params
from web.parser.youdo.youdo_categories import categories_info

import psycopg2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC


locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}


def url_exists_in_db(url, cursor):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM tasks WHERE url=%s)', (url,))
    return cursor.fetchone()[0]


def parse_price(price_str):
    price_type_id = 3

    range_type, max_price = 1, 0
    match = re.search(r"\s*(до|более)?\s*(\d+(?:\s*\d+)*)\s*(?:[-—]\s*(\d+(?:\s*\d+)*))?", price_str)
    if match:
        first_price = match.group(2).replace(" ", "")
        second_price = match.group(3).replace(" ", "") if match.group(3) else None
        max_price = max(int(first_price), int(second_price)) if second_price else int(first_price)
        range_group = match.group(1)

        if range_group == "более":
            range_type = 3
        elif range_group == "до" or second_price:
            range_type = 2

    return max_price, range_type, price_type_id


def get_task_details(task_id):
    api_url = f"https://youdo.com/api/tasks/taskmodel/?taskId={task_id}"
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Ошибка при запросе к API: {response.status_code}")
        return None

    task_data = response.json().get('ResultObject', {}).get('TaskData', {})

    if not task_data:
        print("Данные задачи не найдены.")
        return None

    # Извлекаем необходимые данные из ответа API
    title = task_data.get('Title', '').strip()
    description = task_data.get('Description', '').strip().replace('\n', '<br/>')
    description = re.sub(r'[\u2700-\u27bf]', '', description)
    description = re.sub(r'(<br\/?>\s*){3,}', '<br><br><br>', description)

    # Обработка цены
    price_info = task_data.get('Price', {})
    price_str = price_info.get('PriceInHeader', {}).get('StringFormat', '')
    price, price_range_type, price_type_id = parse_price(price_str)

    # Обработка даты публикации
    published_timestamp = task_data.get('Dates', {}).get('CreationDate', 0)
    published_date = datetime.utcfromtimestamp(published_timestamp // 1000) + timedelta(hours=3)
    published_date = published_date.strftime('%Y-%m-%d %H:%M:%S')

    # Категория и подкатегория
    category = task_data.get('CategoryInfo', {}).get('Name', '').strip()
    subcategory = task_data.get('SubcategoryInfo', {}).get('Name', '').strip()

    is_remote = task_data.get('City', '') is None

    task_details = {
        'url': f"https://youdo.com/t{task_id}",
        'title': title,
        'description': description,
        'price': price,
        'price_range_type': price_range_type,
        'price_type_id': price_type_id,
        'published_date': published_date,
        'standard_category': category,
        'standard_subcategory': subcategory,
        'is_remote': is_remote
    }

    return task_details


task_queue = queue.Queue()
print_lock = threading.Lock()
producer_finished_event = threading.Event()


def producer(retries=5):
    options = webdriver.ChromeOptions()

    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    # Дополнительные опции можно добавить здесь

    # Используем ChromeDriverManager для автоматической установки драйвера
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get('https://youdo.com/tasks-all-opened-all')

    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    empty_count = 0

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'TasksList_list')]"))
        )

        while empty_count < 3:
            new_task_count = 0
            tasks_items = driver.find_elements(By.XPATH, "//*[contains(@class, 'TasksList_list')]//*[contains(@class, "
                                                         "'TasksList_listItem') and not(contains(@class, "
                                                         "'TasksList_banner'))]")
            for task_item in tasks_items[-50:]:
                task_link = task_item.find_elements(By.XPATH, ".//*[contains(@class, 'TasksList_title') and not("
                                                              "contains(@class, 'TasksList_titleBlock')) and not("
                                                              "contains(@class, 'TasksList_titleWrapper'))]")
                task_id = task_link[0].get_attribute('data-id')
                url = f"https://youdo.com/t{task_id}"
                if (not task_id in task_queue.queue) and (not url_exists_in_db(url, cursor)):
                    new_task_count += 1
                    task_queue.put((task_id, 0))

            if new_task_count == 0:
                empty_count += 1
            else:
                empty_count = 0

            try:
                show_more_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//*[contains(@class, 'TasksList_showMoreButton')]"))
                )
                show_more_button.click()
                # Ожидаем появления новых элементов, количество которых больше, чем было
                WebDriverWait(driver, 60).until(
                    lambda d: len(d.find_elements(By.XPATH, "//*[contains(@class, 'TasksList_list')]//*[contains("
                                                            "@class, 'TasksList_listItem') and not(contains(@class, "
                                                            "'TasksList_banner'))]")) > len(tasks_items)
                )
            except Exception as e:
                print("Достигнут конец списка заданий или произошла ошибка.")
                break

    finally:
        cursor.close()
        conn.close()
        driver.quit()
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
        if task_info['is_remote']:
            for subcategory_info in categories_info:
                if task_info['standard_subcategory'] == subcategory_info[1]:
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


def start_youdo_parser():
    global tasks_counter

    producer_finished_event.clear()
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Запускаем рабочие потоки потребителей
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(consumer) for _ in range(1)]

    # Дождемся завершения всех потребителей
    concurrent.futures.wait(futures)

    # Выводим итоговую информацию
    print(f"Обработано заданий: {tasks_counter}")


    tasks_counter = 0


if __name__ == "__main__":
    start_youdo_parser()
