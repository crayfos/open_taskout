import pandas as pd
import string
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

from CNNmodel import TextCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class TextDataset(Dataset):
    def __init__(self, texts, labels, build_vocab=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = get_tokenizer('spacy', 'ru_core_news_sm')
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words('russian'))
        self.vocab = self.build_vocab(self.texts) if build_vocab else self.load_vocab()

    def build_vocab(self, texts, min_freq=3):
        counter = Counter()
        for text in texts:
            tokens = self.preprocess_text(text)
            counter.update(tokens)

        vocab = {'<pad>': 0, '<unk>': 1}

        # Создание словаря, исключая слова низкой с частотой
        vocab.update({word: i + 2 for i, (word, freq) in enumerate(counter.most_common()) if freq >= min_freq})

        with open('vocab.txt', 'w', encoding='utf-8') as f:
            for value, key in vocab.items():
                f.write(f'{key}\t{value}\n')

        return vocab

    def load_vocab(self):
        vocab = {}
        with open('vocab.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    value, key = line.split('\t')
                    if key is not None:
                        vocab.update({str(key): int(value)})
        return vocab

    def preprocess_text(self, text):
        punctuation = string.punctuation + "–" + "…" + "—" + "\t" + "\n" + "‚" + " " + "«" + "»" + "•" + '\u2028'

        tokens = self.tokenizer(text)
        tokens = [token.lower() for token in tokens if token not in punctuation and not all(char in punctuation for char in token)]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def get_text_lengths(self):
        lengths = []
        for text in self.texts:
            tokens = self.preprocess_text(text)
            lengths.append(len(tokens))
        return lengths

    def encode_text(self, text):
        tokens = self.preprocess_text(text)
        max_length = 150
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return torch.tensor([self.vocab.get(token, self.vocab['<unk>']) for token in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.encode_text(self.texts[idx])
        label = self.labels[idx].clone().detach()
        return text, label


df = pd.read_csv('../train_data.csv')
train_texts, train_labels = df['Texts'].tolist(), df['Labels'].tolist()
categories = list(set(train_labels))
num_classes = len(categories)

train_labels_one_hot = torch.zeros((len(train_labels), num_classes))
train_labels_one_hot[range(len(train_labels)), train_labels] = 1
train_dataset = TextDataset(train_texts, train_labels_one_hot, build_vocab=True)


df = pd.read_csv('../test_data.csv')
test_texts, test_labels = df['Texts'].tolist(), df['Labels'].tolist()

test_labels_one_hot = torch.zeros((len(test_labels), num_classes))
test_labels_one_hot[range(len(test_labels)), test_labels] = 1
test_dataset = TextDataset(test_texts, test_labels_one_hot)



import matplotlib.pyplot as plt

lengths = train_dataset.get_text_lengths()
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50)
plt.xlabel('Длина текста')
plt.ylabel('Частотность')
plt.title('Распределение длин текстов')
# plt.show()

most_common_length = max(set(lengths), key=lengths.count)
print(f'Most common text length: {most_common_length}')


# Гиперпараметры модели
vocab_size = len(train_dataset.vocab)
embed_dim = 128
num_classes = len(categories)

num_filters = 256
filter_sizes = [2, 2, 2, 2, 2]

model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, filter_sizes).to(device)

batch_size = 8
learning_rate = 0.001
num_epochs = 10


def collate_batch(batch):
    texts, labels = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels


# DataLoader для обучающей и валидационной выборок
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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

# torch.save(model.state_dict(), 'text_cnn_model.pth')
