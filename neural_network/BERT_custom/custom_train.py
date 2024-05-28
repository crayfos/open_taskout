import pandas as pd
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, AddedToken
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.auto import tqdm, trange
import numpy as np
import re
import torch.nn.functional as F
from collections import Counter, defaultdict

from model_dataset import BertForMultiClassSequenceClassification, TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LABELS = ['разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое']

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

def preprocess_text(text):
    text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
    text = text.lower()
    return text

# Загрузка данных
df_train = pd.read_csv('../train_data.csv')
train_texts, train_labels = df_train['Texts'].tolist(), df_train['Labels'].tolist()

df_test = pd.read_csv('../test_data.csv')
test_texts, test_labels = df_test['Texts'].tolist(), df_test['Labels'].tolist()

all_texts = train_texts

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

# Топ-10 самых частых слов
top_10_words = sorted(split_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 most common split words:")
for word, freq in top_10_words:
    print(f"{word}: {freq}")

# Выбираем только те слова, которые встречаются более 10 раз
most_common_split_words = [word for word, freq in split_word_freq.items() if freq > 30]

# Добавление самых популярных слов в словарь токенизатора с атрибутом single_word
new_tokens = [AddedToken(word, single_word=True) for word in most_common_split_words]
num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
print(f"Added {num_added_tokens} tokens")

model = BertForMultiClassSequenceClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=6).to(device)

model.resize_token_embeddings(len(tokenizer))

# Инициализация эмбеддингов новых слов
def initialize_new_embeddings(model, tokenizer, new_words):
    with torch.no_grad():
        # Получение текущих эмбеддингов
        current_embeddings = model.get_input_embeddings().weight.data

        for word in new_words:
            word_tokens = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            token_embeddings = current_embeddings[token_ids, :]
            new_embedding = token_embeddings.mean(dim=0)
            new_word_id = tokenizer.convert_tokens_to_ids(word)
            current_embeddings[new_word_id, :] = new_embedding

# Инициализация эмбеддингов для новых слов
initialize_new_embeddings(model, tokenizer, most_common_split_words)


model.config.label2id = {label: i for i, label in enumerate(LABELS)}
model.config.id2label = {i: label for i, label in enumerate(LABELS)}


train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

train_data, val_data = train_dataset, test_dataset

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def calculate_f1(model, dataloader):
    model.eval()
    facts, preds = predict_with_model(model, dataloader)
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(facts, preds, average='weighted')
    return f1

def calculate_validation_loss(model, validation_dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs
            total_loss += loss.item()
    return total_loss / len(validation_dataloader)

def predict_with_model(model, dataloader, verbose=False):
    preds = []
    facts = []

    tq = dataloader
    if verbose:
        tq = tqdm(dataloader)

    for batch in tq:
        labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        facts.append(labels.cpu().numpy())

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            preds.append(F.softmax(logits, dim=-1).cpu().numpy())

    facts = np.concatenate(facts)
    preds = np.concatenate(preds)
    return facts, preds

batch_size = 16

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=collate_fn)
dev_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Freeze BERT encoder and train only classifier
model.freeze_bert_encoder()

optimizer = torch.optim.AdamW(params=model.classifier.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .5 ** epoch)

for epoch in trange(1):
    model.train()
    cleanup()

    print('LR:', scheduler.get_last_lr())
    tq = tqdm(train_dataloader)
    for i, batch in enumerate(tq):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tq.set_description(f'loss: {loss.item():.4f}')

    val_loss = calculate_validation_loss(model, dev_dataloader)
    scheduler.step()

    f1 = calculate_f1(model, dev_dataloader)
    print(f'\n[epoch {epoch + 1}] val f1: {f1:.4f}, val loss: {val_loss:.4f}\n\n')

# Unfreeze BERT encoder and continue training entire model
model.unfreeze_bert_encoder()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .7 ** epoch)

for epoch in trange(1, 3):
    model.train()
    cleanup()

    print('LR:', scheduler.get_last_lr())
    tq = tqdm(train_dataloader)
    for i, batch in enumerate(tq):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs
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
