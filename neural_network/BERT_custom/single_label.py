import pandas as pd
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
import numpy as np
import re
from collections import Counter, defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LABELS = ['разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

# Функция для предварительной обработки текста
def preprocess_text(text):
    text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
    text = text.lower()
    return text

# Загрузка данных
df_train = pd.read_csv('../train_data.csv')
train_texts, train_labels = df_train['Texts'].tolist(), df_train['Labels'].tolist()

df_test = pd.read_csv('../test_data.csv')
test_texts, test_labels = df_test['Texts'].tolist(), df_test['Labels'].tolist()

all_texts = train_texts + test_texts

# Функция для нахождения слов, разбитых на несколько частей
def find_split_words(texts, tokenizer):
    split_word_freq = defaultdict(int)
    for text in texts:
        text = preprocess_text(text)
        tokens = tokenizer.tokenize(text)
        current_word_parts = []
        for token in tokens:
            if token.startswith("##"):
                current_word_parts.append(token)
            else:
                if len(current_word_parts) > 1:
                    full_word = "".join([part.replace("##", "") for part in current_word_parts])
                    split_word_freq[full_word] += 1
                current_word_parts = [token] if not token.startswith("##") else [token]
        if len(current_word_parts) > 1:
            full_word = "".join([part.replace("##", "") for part in current_word_parts])
            split_word_freq[full_word] += 1

    return split_word_freq

# Находим разбитые на части слова
split_word_freq = find_split_words(all_texts, tokenizer)

# Выбираем только те слова, которые встречаются более 10 раз
most_common_split_words = [word for word, freq in split_word_freq.items() if freq > 20]

# Топ-10 самых частых слов
top_10_words = sorted(split_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 most common split words:")
for word, freq in top_10_words:
    print(f"{word}: {freq}")

# Добавление самых популярных слов в словарь токенизатора
num_added_tokens = tokenizer.add_tokens(most_common_split_words)
print(f"Added {num_added_tokens} tokens")

# Инициализация модели с учетом нового словаря
model = BertForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2",
    num_labels=len(LABELS),
    problem_type='single_label_classification'
).to(device)
model.resize_token_embeddings(len(tokenizer))

# Инициализация эмбеддингов новых слов
def initialize_new_embeddings(model, tokenizer, new_words):
    with torch.no_grad():
        # Получение текущих эмбеддингов
        current_embeddings = model.get_input_embeddings().weight.data
        embedding_dim = current_embeddings.shape[1]

        for word in new_words:
            word_tokens = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            token_embeddings = current_embeddings[token_ids, :]
            new_embedding = token_embeddings.mean(dim=0)
            new_word_id = tokenizer.convert_tokens_to_ids(word)
            current_embeddings[new_word_id, :] = new_embedding

# Инициализация эмбеддингов для новых слов
initialize_new_embeddings(model, tokenizer, most_common_split_words)

model.config.label2id = label2id
model.config.id2label = id2label

# Датасет
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        text = preprocess_text(text)
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Создание датасетов
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

# Функция очистки
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

cleanup()

# Функция для вычисления F1-метрики
def calculate_f1(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1

# Функция для вычисления потерь на валидации
def calculate_validation_loss(model, validation_dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(validation_dataloader)

# Параметры обучения
batch_size = 16

# Функция для объединения батчей
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Создание DataLoader'ов
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=collate_fn)
dev_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Вычисление начального F1
f1 = calculate_f1(model, dev_dataloader)
print(f'\n[epoch 0] val f1: {f1:.4f}\n\n')

# Инициализация оптимизатора и scheduler'а
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.4 ** epoch)


for param in model.parameters():
    param.requires_grad = True

# Обучение модели
for epoch in trange(4):
    model.train()
    cleanup()

    print('LR:', scheduler.get_last_lr())
    tq = tqdm(train_dataloader)
    for i, batch in enumerate(tq):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tq.set_description(f'loss: {loss.item():.4f}')

    val_loss = calculate_validation_loss(model, dev_dataloader)
    scheduler.step()

    f1 = calculate_f1(model, dev_dataloader)
    print(f'\n[epoch {epoch + 1}] val f1: {f1:.4f}, val loss: {val_loss:.4f}\n\n')

    model_path = "./freelance_bert"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f'\n[epoch {epoch + 1}] model saved')
