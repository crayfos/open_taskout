import pandas as pd
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from IPython.display import display
import string
import re
from torch.nn import BCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


LABELS = {'разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое'}

# RuBERT Tiny токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = BertForSequenceClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=6,
                                                      problem_type='multi_label_classification').to(device)
model.config.label2id = {label: i for i, label in enumerate(LABELS)}
model.config.id2label = {i: label for i, label in enumerate(LABELS)}
# model.loss_fct = BCEWithLogitsLoss()

# Датасет
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.texts)

    def preprocess_text(self, text):
        text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
        text = text.lower()

        # text = ''.join([char.lower() for char in text])
        return text

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        text = self.preprocess_text(text)
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        label_tensor = torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).float()

        return {'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': label_tensor}


# Загрузка данных
df = pd.read_csv('../train_data.csv')
train_texts, train_labels = df['Texts'].tolist(), df['Labels'].tolist()
df = pd.read_csv('../test_data.csv')
test_texts, test_labels = df['Texts'].tolist(), df['Labels'].tolist()
categories = list(set(train_labels))
num_classes = len(categories)

train_dataset = TextDataset(train_texts, train_labels, tokenizer, model)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, model)

train_data, val_data = train_dataset, test_dataset


# Cleaning unnecessary data during training
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


cleanup()


def calculate_f1(model, dataloader):
    model.eval()
    facts, preds = predict_with_model(model, dataloader)
    preds = np.argmax(preds, axis=1)
    facts = np.argmax(facts, axis=1)
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
            loss = outputs.loss
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

        # Transfer to the appropriate device (e.g., GPU)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        facts.append(labels.cpu().numpy())

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.append(torch.softmax(logits, dim=-1).cpu().numpy())

    facts = np.concatenate(facts)
    preds = np.concatenate(preds)
    return facts, preds


batch_size = 32


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
dev_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

f1 = calculate_f1(model, dev_dataloader)
print(f'\n[epoch 0] val f1: {f1:.4f}\n\n')

optimizer = torch.optim.AdamW(params=model.parameters(), lr=4e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .6 ** epoch)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)


for epoch in trange(3):
    model.train()
    cleanup()

    print('LR:', scheduler.get_last_lr())
    tq = tqdm(train_dataloader)
    for i, batch in enumerate(tq):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Update weights
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
