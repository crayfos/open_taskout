import gc
import torch
import re
import pandas as pd

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def load_data(filepath):
    df = pd.read_csv(filepath)
    texts, labels = df['Texts'].tolist(), df['Labels'].tolist()
    return texts, labels


def preprocess_text(text):
    text = re.sub(r'\xa0|&nbsp;|\n|\t|\u2028', ' ', text)
    text = text.lower()
    return text

