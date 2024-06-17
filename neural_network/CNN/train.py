import torch
import matplotlib.pyplot as plt
from web.neural_network.CNN.utils.preprocessing import load_data, encode_labels
from web.neural_network.CNN.utils.dataset import TextDataset
from web.neural_network.CNN.utils.training import train_model, collate_batch
from web.neural_network.CNN.utils.evaluation import evaluate_model, print_classification_report
from web.neural_network.CNN.model.CNNmodel import TextCNN
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_texts, train_labels = load_data('../train_data.csv')
test_texts, test_labels = load_data('../test_data.csv')

categories = list(set(train_labels))
num_classes = len(categories)

train_labels_one_hot = encode_labels(train_labels, num_classes)
test_labels_one_hot = encode_labels(test_labels, num_classes)


train_dataset = TextDataset(train_texts, train_labels_one_hot, build_vocab=True)
test_dataset = TextDataset(test_texts, test_labels_one_hot)


# Визуализация распределения длин текстов
lengths = train_dataset.get_text_lengths()
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50)
plt.xlabel('Длина текста')
plt.ylabel('Частотность')
plt.title('Распределение длин текстов')
plt.show()

most_common_length = max(set(lengths), key=lengths.count)
print(f'Most common text length: {most_common_length}')


vocab_size = len(train_dataset.vocab)
embed_dim = 128
num_classes = len(categories)
num_filters = 256
filter_sizes = [2, 2, 2, 2, 2]

model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, filter_sizes).to(device)

batch_size = 8
learning_rate = 0.001
num_epochs = 8

# DataLoader для обучающей и валидационной выборок
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

# Тренировка модели
train_model(model, train_loader, val_loader, num_epochs, device, learning_rate)

# Оценка модели
val_loss, val_accuracy, val_f1, all_labels, all_preds = evaluate_model(model, val_loader, device)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}')

# Вывод подробного отчёта
print_classification_report(all_labels, all_preds, num_classes)


torch.save(model.state_dict(), 'text_cnn_model.pth')
