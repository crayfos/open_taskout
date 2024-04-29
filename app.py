from flask import Flask, render_template, jsonify, request
import psycopg2
from db_config import db_params

from dateutil import parser
import arrow

from flask_apscheduler import APScheduler
from datetime import datetime, timedelta
from threading import Lock

from web.parser.parser import start_parser

app = Flask(__name__)
scheduler = APScheduler()


def humanize_datetime(datetime_str):
    dt = parser.parse(str(datetime_str))
    return arrow.get(dt).humanize(locale='ru')


def process_complaints():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    # Получаем все нерассмотренные жалобы
    cur.execute(
        "SELECT c.complaint_id, tp.processing_id, c.user_proposed_category, tp.category "
        "FROM complaints c "
        "JOIN task_processing tp ON c.processing_id = tp.processing_id "
        "WHERE c.complaint_status = 0"
    )
    complaints = cur.fetchall()

    for complaint_id, processing_id, user_proposed_category, current_category in complaints:
        if user_proposed_category != current_category:
            cur.execute(
                "UPDATE task_processing SET status = %s "
                "WHERE processing_id = %s",
                (4, processing_id)
            )
            complaint_status = 2
        else:
            cur.execute(
                "UPDATE task_processing SET category = %s, status = %s, category_change_date = %s "
                "WHERE processing_id = %s",
                (user_proposed_category, 4, datetime.now(), processing_id)
            )
            complaint_status = 1

        cur.execute(
            "UPDATE complaints SET complaint_status = %s "
            "WHERE complaint_id = %s",
            (complaint_status, complaint_id)
        )
    conn.commit()
    conn.close()


def get_categories():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute(
        "SELECT categories.category_id, categories.category_name, COUNT(tasks.task_id), "
        "(SELECT COUNT(*) FROM task_processing tp WHERE tp.category = categories.category_id AND tp.status = 4) AS "
        "status_4_task_count,"
        "(SELECT COUNT(task_processing.task_id) FROM task_processing) AS total_task_count "
        "FROM categories "
        "LEFT JOIN task_processing ON categories.category_id = task_processing.category "
        "LEFT JOIN tasks ON tasks.task_id = task_processing.task_id "
        "GROUP BY categories.category_id "
        "ORDER BY categories.category_id ASC "
    )
    categories = cur.fetchall()
    conn.close()
    return categories


def get_tasks(category, page):
    process_complaints()
    tasks_per_page = 30
    offset = (page - 1) * tasks_per_page
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    if category == 'all':
        cur.execute(
            "SELECT tasks.*, statuses.status_name, categories.category_name FROM tasks "
            "JOIN task_processing ON tasks.task_id = task_processing.task_id "
            "JOIN categories ON categories.category_id = task_processing.category "
            "JOIN statuses ON task_processing.status = statuses.status_id "
            "ORDER BY tasks.published_date DESC "
            "LIMIT %s OFFSET %s", (tasks_per_page, offset))
    else:
        cur.execute(
            "SELECT tasks.*, statuses.status_name, categories.category_name FROM tasks "
            "JOIN task_processing ON tasks.task_id = task_processing.task_id "
            "JOIN categories ON categories.category_id = task_processing.category "
            "JOIN statuses ON task_processing.status = statuses.status_id "
            "WHERE task_processing.category = %s "
            "ORDER BY tasks.published_date DESC "
            "LIMIT %s OFFSET %s", (category, tasks_per_page, offset))
    tasks = cur.fetchall()

    tasks_list = []
    for task in tasks:
        task_dict = {
            'task_id': task[0],
            'task_link': task[1],
            'title': task[2],
            'description': task[3],
            'price': task[4],
            'price_range_type': task[5],
            'price_type': task[6],
            'published_date': humanize_datetime(task[7]),
            'status_name': task[10],
            'category_name': task[11],
        }
        cur.execute("SELECT * FROM complaints WHERE processing_id = %s", (task_dict['task_id'],))
        task_dict['complaint_exists'] = cur.fetchone() is not None
        tasks_list.append(task_dict)

    conn.close()
    return tasks_list


@app.route('/<category>/')
@app.route('/<category>/<int:page>')
def category_index(category, page=1):
    categories = get_categories()
    tasks = get_tasks(category, page)
    return render_template('index.html', tasks=tasks, categories=categories, current_category=category,
                           current_page=page)


@app.route('/<int:page>')
def index_page(page):
    return category_index('all', page)


@app.route('/')
def index():
    return category_index('all', 1)


@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    processing_id = request.form.get('processing_id')
    user_proposed_category = request.form.get('user_proposed_category')
    complaint_date = datetime.now()

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("SELECT * FROM complaints WHERE processing_id = %s", (processing_id,))
    existing_complaint = cur.fetchone()

    if existing_complaint is None:
        cur.execute(
            "INSERT INTO complaints (processing_id, user_proposed_category, complaint_date) "
            "VALUES (%s, %s, %s)",
            (processing_id, user_proposed_category, complaint_date)
        )
    else:
        cur.execute(
            "UPDATE complaints SET user_proposed_category = %s, complaint_date = %s "
            "WHERE processing_id = %s",
            (user_proposed_category, complaint_date, processing_id)
        )
    conn.commit()
    conn.close()
    return jsonify({'message': "Жалоба отправлена."})


parser_lock = Lock()


def scheduled_parsing():
    with parser_lock:
        print("Запуск парсера...")
        start_parser()

        next_parsing_time = datetime.now() + timedelta(minutes=1)
        scheduler.add_job(id='Scheduled Parsing', func=scheduled_parsing, trigger='date', run_date=next_parsing_time)


def scheduled_neural_network():
    print("Запуск нейросети...")


scheduler.init_app(app)
scheduler.start()

if __name__ == '__main__':
    if not scheduler.get_job('Scheduled Parsing'):
        scheduler.add_job(id='Scheduled Parsing', func=scheduled_parsing, trigger='date',
                          run_date=datetime.now() + timedelta(minutes=1))
    app.run()
