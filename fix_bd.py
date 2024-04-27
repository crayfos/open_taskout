from web.parser.db_config import db_params
from web.parser.habr.habr_categories import categories_info
import psycopg2


def check_and_update_complaints_status(categories_info):
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # Создаем словарь для быстрого доступа к категориям по их названиям
    category_to_id = {(info[1], info[2]): info[0] for info in categories_info}

    # Получаем все таски со статусом 4 (manual)
    cur.execute(
        "SELECT tp.processing_id, tp.category, c.complaint_id, t.standard_category, t.standard_subcategory "
        "FROM task_processing tp "
        "JOIN complaints c ON tp.processing_id = c.processing_id "
        "JOIN tasks t ON t.task_id = tp.task_id "
        "WHERE tp.status = 4 AND c.complaint_status = 1"
    )
    tasks = cur.fetchall()

    for processing_id, current_category_id, complaint_id, standard_category, standard_subcategory in tasks:

        if category_to_id[(standard_category,standard_subcategory)] != current_category_id:
            # Устанавливаем статус жалобы в 2, если категория не соответствует
            cur.execute(
                "UPDATE complaints SET complaint_status = 2 WHERE complaint_id = %s",
                (complaint_id,)
            )

    conn.commit()
    conn.close()


# Вызов функции для проверки и обновления статусов жалоб
check_and_update_complaints_status(categories_info)
