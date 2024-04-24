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

from habr_categories import categories_info

locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')


def get_task_links(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        tasks = soup.find_all('article', class_='task')

        task_links = []
        for task in tasks:
            title_element = task.find('div', class_='task__title')
            link = 'https://freelance.habr.com' + title_element.find('a')['href']
            task_links.append(link)
        return task_links
    except (Timeout, ConnectionError) as e:
        print(f"Возникла ошибка при подключении к {url}: {e}")
    except RequestException as e:
        print(f"Произошла ошибка при запросе к {url}: {e}")
    return ['error']


def get_published_date(meta_string):
    # Словарь для замены русского названия месяца на число
    months = {
        'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
        'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
        'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
    }

    date_match = re.search(r'\d{1,2} .+ \d{4}, \d{2}:\d{2}', meta_string)
    if date_match:
        # Получаем строку даты из регулярного выражения
        published_date_str = date_match.group(0)

        # Заменяем русское название месяца на численное значение
        for rus_month, num_month in months.items():
            if rus_month in published_date_str:
                published_date_str = published_date_str.replace(rus_month, num_month)
                break

        # Преобразуем строку даты в объект datetime
        published_date = datetime.strptime(published_date_str, '%d %m %Y, %H:%M')
        # Преобразуем в строку в формате ISO для PostgreSQL
        return published_date.strftime('%Y-%m-%d %H:%M:%S')
    return None

def parse_price(price_str):
    price_type_id = 1  # Договорная
    if "руб." in price_str:
        if "за проект" in price_str:
            price_type_id = 3  # За проект
        elif "за час" in price_str:
            price_type_id = 2  # За час

    price_str = re.sub(r'[^0-9]', '', price_str)  # Удалить все нецифровые символы
    price = int(price_str) if price_str else None
    return price, price_type_id


def get_task_details(task_url):
    try:
        response = requests.get(task_url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('h2', class_='task__title').get_text(separator=' ', strip=True)
        price_str = soup.find('div', class_='task__finance').get_text(strip=True)
        price, price_type_id = parse_price(price_str)

        meta = soup.find('div', class_='task__meta').get_text(strip=True)
        published_date = get_published_date(meta)

        description = str(soup.find('div', class_='task__description'))\
            .replace('<div class="task__description">\n', '')\
            .replace('\n</div>', '')



        task_details = {
            'url': task_url,
            'title': title,
            'description': description,
            'price': price if price else 'Договорная',
            'price_type_id': price_type_id,
            'published_date': published_date,
            'standard_category': '',
            'standard_subcategory': ''
        }

        return task_details
    except ConnectionError as e:
        print(f"Ошибка соединения при запросе к {task_url}: {e}")


task_queue = queue.Queue()
print_lock = threading.Lock()


def producer(categories_info, max_retries=5):
    retries = {url: 0 for deep_category, category, subcategory, url in categories_info}
    page_numbers = {url: 1 for deep_category, category, subcategory, url in categories_info}

    while categories_info:
        for deep_category, category, subcategory, url in list(categories_info):
            page_num = page_numbers[url]
            if retries[url] < max_retries:
                links = get_task_links(f"{url}&page={page_num}")
                if links and links[0] != 'error':
                    for link in links:
                        task_queue.put((link, category, subcategory, 0))
                    page_numbers[url] += 1
                elif links and links[0] == 'error':
                    retries[url] += 1
                    print(f"Повторная попытка {retries[url]} для URL {url}")
                else:
                    categories_info.remove((category, subcategory, url))
            else:
                print(f"Превышено максимальное количество попыток для {url}")
                categories_info.remove((category, subcategory, url))


# Функция consumer извлекает данные из очереди и обрабатывает каждую задачу
def consumer(retry_limit=3, delay_between_retries=5):
    while True:
        try:
            link, category, subcategory, retries = task_queue.get(timeout=10)

            # Попытаемся получить детали задания
            task_info = get_task_details(link)
            task_info['standard_category'] = category
            task_info['standard_subcategory'] = subcategory

            # Выводим и сохраняем информацию о задании
            with print_lock:
                print(task_info)
            tasks_data.append(task_info)

        except queue.Empty:
            time.sleep(10)
            if task_queue.empty():
                return

        except Exception as e:
            with print_lock:
                print(f"Ошибка при обработке ссылки {link}: {e}, попытка {retries}")
            retries += 1
            if retries <= retry_limit:
                # Планируем повторную попытку с задержкой
                time.sleep(delay_between_retries)
                task_queue.put((link, category, subcategory, retries))
            else:
                with print_lock:
                    print(f"Достигнут лимит попыток для ссылки {link}")
        finally:
            task_queue.task_done()

if __name__ == "__main__":
    producer_thread = threading.Thread(target=producer, args=(categories_info,))
    producer_thread.start()

    tasks_data = []

    # Запускаем рабочие потоки потребителей
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(consumer) for _ in range(10)]

    # Дождемся завершения всех потребителей
    concurrent.futures.wait(futures)

    # Выводим итоговую информацию
    print(f"Обработано {len(tasks_data)} заданий.")
