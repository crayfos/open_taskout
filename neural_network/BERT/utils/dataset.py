import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from web.neural_network.BERT2.utils import utils
from web.neural_network.BERT2 import config

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        text = utils.preprocess_text(text)
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        label_tensor = torch.nn.functional.one_hot(torch.tensor(label), num_classes=len(config.LABELS)).float()

        return {'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': label_tensor}


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def get_dataloader(batch_size, train_dataset, test_dataset):
    train_dataloader = None
    if train_dataset is not None:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=collate_fn)

    dev_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_fn)
    return train_dataloader, dev_dataloader


