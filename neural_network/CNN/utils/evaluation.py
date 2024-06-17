import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn

def evaluate_model(model, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()

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

    return val_loss / len(val_loader), val_accuracy, val_f1, all_preds, all_labels


def print_classification_report(all_labels, all_preds, num_classes):
    target_names = [f'Class {i}' for i in range(num_classes)]
    print(classification_report(all_labels, all_preds, target_names=target_names))
