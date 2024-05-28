import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import re
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model_dataset import BertForMultiClassSequenceClassification, TextDataset
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_path = "./freelance_bert"
model = BertForMultiClassSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)

# Collate function to pad sequences dynamically
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Load test data
df = pd.read_csv('../test_data.csv')
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
            logits = outputs

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
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

df = pd.DataFrame({
    'Text': texts,
    'True Label': [f"{model.config.id2label[label]} - ({conf*100:.2f}%)" for label, conf in zip(true_labels, true_confidences)],
    'Predicted': [f"{model.config.id2label[pred]} - ({conf*100:.2f}%)" for pred, conf in zip(predictions, pred_confidences)],
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
