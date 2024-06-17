import pandas as pd
import torch

def load_data(filepath):
    df = pd.read_csv(filepath)
    texts, labels = df['Texts'].tolist(), df['Labels'].tolist()
    return texts, labels

def encode_labels(labels, num_classes):
    labels_one_hot = torch.zeros((len(labels), num_classes))
    labels_one_hot[range(len(labels)), labels] = 1
    return labels_one_hot
