import pandas as pd
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LABELS = ['разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# RuBERT Tiny токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = BertForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2",
    num_labels=len(LABELS),
    problem_type='single_label_classification'
).to(device)
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

    def preprocess_text(self, text):
        text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
        text = text.lower()
        return text

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        text = self.preprocess_text(text)
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Загрузка данных
df_train = pd.read_csv('../train_data.csv')
train_texts, train_labels = df_train['Texts'].tolist(), df_train['Labels'].tolist()

df_test = pd.read_csv('../test_data.csv')
test_texts, test_labels = df_test['Texts'].tolist(), df_test['Labels'].tolist()

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)


# Cleaning unnecessary data during training
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


cleanup()


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


batch_size = 16


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
dev_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

f1 = calculate_f1(model, dev_dataloader)
print(f'\n[epoch 0] val f1: {f1:.4f}\n\n')

optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.4 ** epoch)

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
