import pandas as pd

import string
from razdel import tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchtext.data.utils import get_tokenizer

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from RNNmodel import AttentionRNN
from CNNmodel import TextCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None):
        self.texts = texts
        self.labels = labels
        # self.tokenizer = tokenize
        self.tokenizer = get_tokenizer('spacy', 'ru_core_news_sm')
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words('russian'))
        self.vocab = vocab or self.build_vocab(self.texts)
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

    def build_vocab(self, texts):
        # Подсчёт слов для создания словаря
        counter = Counter()
        for text in texts:
            tokens = self.tokenizer(text)
            # tokens = [word.text for word in tokens]
            counter.update(token.lower() for token in tokens if token not in string.punctuation)
        return {word: i + 2 for i, (word, _) in enumerate(counter.most_common())}

    def preprocess_text(self, text):
        # Предобработка текста
        tokens = self.tokenizer(text)
        # tokens = [word.text for word in tokens]
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def encode_text(self, text):
        # Кодирование текста в последовательность индексов
        tokens = self.preprocess_text(text)
        return torch.tensor([self.vocab.get(token, 1) for token in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.encode_text(self.texts[idx])
        label = self.labels[idx].clone().detach()
        return text, label


df = pd.read_csv('../data.csv')
texts = df['Texts'].tolist()
labels = df['Labels'].tolist()
categories = list(set(labels))

# Преобразование меток в one-hot encoding
num_classes = len(categories)
labels_one_hot = torch.zeros((len(labels), num_classes))
labels_one_hot[range(len(labels)), labels] = 1

dataset = TextDataset(texts, labels_one_hot)


# Гиперпараметры модели
vocab_size = len(dataset.vocab) + 2  # +2 for <pad> and <unk> tokens
embed_dim = 128  # размерность векторных представлений слов
num_classes = len(categories)  # количество классов (категорий)

num_filters = 256
filter_sizes = [2, 3, 4]

num_layers = 5  # количество слоев RNN
bidirectional = True  # использовать двунаправленную RNN
dropout = 0  # вероятность применения dropout

# Создание экземпляра модели
# model = AttentionRNN(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, filter_sizes).to(device)

batch_size = 8
learning_rate = 0.001
num_epochs = 5

# Разделение датасета
train_data, val_data = train_test_split(dataset, test_size=0.1)


def collate_batch(batch):
    texts, labels = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels


# DataLoader для обучающей и валидационной выборок
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_batch)


# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()  # Обновление скорости обучения

    # Валидация
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.argmax(all_labels, axis=1)
    all_preds = np.argmax(all_preds, axis=1)

    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

# Вывод подробного отчета о классификации
print(classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(num_classes)]))

# Сохранение модели (опционально)
# torch.save(model.state_dict(), 'text_cnn_model.pth')
