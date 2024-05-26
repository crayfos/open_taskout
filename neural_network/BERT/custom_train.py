import pandas as pd
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.auto import tqdm, trange
import numpy as np
import re
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LABELS = {'разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое'}

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

class BertForMultiClassSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=6):
        super(BertForMultiClassSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

model = BertForMultiClassSequenceClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=6).to(device)
model.config.label2id = {label: i for i, label in enumerate(LABELS)}
model.config.id2label = {i: label for i, label in enumerate(LABELS)}

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
            'labels': torch.tensor(label).long()
        }

# Загрузка данных
df = pd.read_csv('../train_data.csv')
train_texts, train_labels = df['Texts'].tolist(), df['Labels'].tolist()
df = pd.read_csv('../test_data.csv')
test_texts, test_labels = df['Texts'].tolist(), df['Labels'].tolist()

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

batch_size = 32

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=collate_fn)
dev_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Freeze BERT encoder and train only classifier
model.freeze_bert_encoder()

optimizer = torch.optim.AdamW(params=model.classifier.parameters(), lr=4e-4)
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

optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .7 ** epoch)

for epoch in trange(1, 6):
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
