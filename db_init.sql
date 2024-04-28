CREATE TABLE price_types (
    price_type_id SERIAL PRIMARY KEY,
    price_type_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE tasks (
    task_id SERIAL PRIMARY KEY,
    url VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    price INT,
    price_range_type INT DEFAULT 0,
    price_type_id INT REFERENCES price_types(price_type_id),
    published_date TIMESTAMP WITH TIME ZONE NOT NULL,
    standard_category VARCHAR(100),
    standard_subcategory VARCHAR(100)
);

CREATE TABLE statuses (
    status_id SERIAL PRIMARY KEY,
    status_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE task_processing (
    processing_id SERIAL PRIMARY KEY,
    task_id INT UNIQUE NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    category INT REFERENCES categories(category_id),
    status INT NOT NULL REFERENCES statuses(status_id),
    category_change_date TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE complaints (
    complaint_id SERIAL PRIMARY KEY,
    processing_id INT NOT NULL REFERENCES task_processing(processing_id),
    user_proposed_category INT REFERENCES categories(category_id),
    complaint_status smallint NOT NULL DEFAULT 0,
    complaint_date TIMESTAMP WITH TIME ZONE NOT NULL
);


INSERT INTO price_types (price_type_name) VALUES ('договорная'), ('за час'), ('за проект');
INSERT INTO statuses (status_name) VALUES ('unfilled'), ('automatic'), ('neural_network'), ('manual');
INSERT INTO categories (category_name) VALUES ('Разработка'), ('Дизайн'), ('Контент'), ('Маркетинг'), ('Бизнес'), ('Другое');