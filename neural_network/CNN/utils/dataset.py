import torch
from torch.utils.data import Dataset
from collections import Counter
from torchtext.data.utils import get_tokenizer
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class TextDataset(Dataset):
    def __init__(self, texts, labels, build_vocab=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = get_tokenizer('spacy', 'ru_core_news_sm')
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words('russian'))
        self.vocab = self.build_vocab(self.texts) if build_vocab else self.load_vocab()

    def build_vocab(self, texts, min_freq=5):
        counter = Counter()
        for text in texts:
            tokens = self.preprocess_text(text)
            counter.update(tokens)

        vocab = {'<pad>': 0, '<unk>': 1}
        vocab.update({word: i + 2 for i, (word, freq) in enumerate(counter.most_common()) if freq >= min_freq})

        with open('vocab.txt', 'w', encoding='utf-8') as f:
            for value, key in vocab.items():
                f.write(f'{key}\t{value}\n')

        return vocab

    def load_vocab(self):
        vocab = {}
        with open('vocab.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    value, key = line.split('\t')
                    if key is not None:
                        vocab.update({str(key): int(value)})
        return vocab

    def preprocess_text(self, text):
        punctuation = string.punctuation + "–" + "…" + "—" + "\t" + "\n" + "‚" + " " + "«" + "»" + "•" + '\u2028'

        tokens = self.tokenizer(text)
        tokens = [token.lower() for token in tokens if token not in punctuation and not all(char in punctuation for char in token)]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def get_text_lengths(self):
        lengths = []
        for text in self.texts:
            tokens = self.preprocess_text(text)
            lengths.append(len(tokens))
        return lengths

    def encode_text(self, text):
        tokens = self.preprocess_text(text)
        max_length = 150
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return torch.tensor([self.vocab.get(token, self.vocab['<unk>']) for token in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.encode_text(self.texts[idx])
        label = self.labels[idx].clone().detach()
        return text, label
