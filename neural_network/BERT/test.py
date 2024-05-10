import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_path = "./freelance_bert"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)


# Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def preprocess_text(self, text):
        text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
        text = text.lower()

        # text = ''.join([char.lower() for char in text])
        return text

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = self.preprocess_text(text)
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Collate function to pad sequences dynamically
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# Load test data
df = pd.read_csv('../train_data.csv')
test_texts = df['Texts'].tolist()
test_labels = df['Labels'].tolist()
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    predictions = []
    true_confidences = []
    pred_confidences = []
    texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, preds = torch.max(probs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            true_confidences.extend(probs[range(len(labels)), labels].cpu().numpy())
            pred_confidences.extend(probs[range(len(labels)), preds].cpu().numpy())
            texts.extend([tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids])

    return true_labels, predictions, true_confidences, pred_confidences, texts

true_labels, predictions, true_confidences, pred_confidences, texts = evaluate_model(model, test_dataloader)

report = classification_report(true_labels, predictions)
print(report)

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()



df = pd.DataFrame({
    'Text': texts,
    'True Label': [f"{label+1} - ({conf*100:.2f}%)" for label, conf in zip(true_labels, true_confidences)],
    'Predicted': [f"{pred+1} - ({conf*100:.2f}%)" for pred, conf in zip(predictions, pred_confidences)],
})

errors_df = df[df['True Label'] != df['Predicted']]
table = PrettyTable()
table.align = "l"
table.hrules = 1

table.max_width["Text"] = 100
table.field_names = ["True Label", "Predicted", "Text"]

for index, row in errors_df.iterrows():
    table.add_row([row['True Label'], row['Predicted'], row['Text']])

# Print the table
print(table)
