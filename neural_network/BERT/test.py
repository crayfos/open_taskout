import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from transformers import AutoTokenizer, BertForSequenceClassification

from utils import utils, tokens, dataset, evaluation
import config


model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH).to(config.device)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

# тест полученного токенизатора
orig_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
examples = [
    "помощь в интеграции яндекс маркета здравствуйте! не получается сделать интеграцию. нужна помощь, тз прилагаю"
]

for example in examples:
    print("Original text:", example)
    print("Tokenized text:", tokenizer.tokenize(example))
    print("Orig tokenized text:", orig_tokenizer.tokenize(example))
    print()

# Загрузка данных
test_texts, test_labels = utils.load_data('../test_data.csv')
test_dataset = dataset.TextDataset(test_texts, test_labels, tokenizer, model)

_, test_dataloader = dataset.get_dataloader(16, None, test_dataset)

# Получение предсказаний
true_labels, predictions, texts = evaluation.evaluate_model(model, test_dataloader, tokenizer)

true_confidences = []
pred_confidences = []

true_labels = np.argmax(true_labels, axis=1)

# Вычисление вероятностей
for true_label, pred in zip(true_labels, predictions):
    true_confidences.append(pred[true_label])
    pred_confidences.append(np.max(pred))

predictions = np.argmax(predictions, axis=1)


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
    'True Label': [f"{label + 1} - ({conf * 100:.2f}%)" for label, conf in zip(true_labels, true_confidences)],
    'Predicted': [f"{pred + 1} - ({conf * 100:.2f}%)" for pred, conf in zip(predictions, pred_confidences)],
})

errors_df = df[df['True Label'] != df['Predicted']]
table = PrettyTable()
table.align = "l"
table.hrules = 1

table.max_width["Text"] = 100
table.field_names = ["True Label", "Predicted", "Text"]

for index, row in errors_df.iterrows():
    table.add_row([row['True Label'], row['Predicted'], row['Text']])

print(table)
