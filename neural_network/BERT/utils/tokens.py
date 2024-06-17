from collections import defaultdict
import torch
from web.neural_network.BERT2.utils import utils


# Функция для нахождения слов, разбитых на несколько частей
def find_split_words(texts, tokenizer):
    split_word_freq = defaultdict(int)
    for text in texts:
        text = utils.preprocess_text(text)
        tokens = tokenizer.tokenize(text)
        current_word_parts = []
        for token in tokens:
            if token.startswith("##"):
                current_word_parts.append(token)
            else:
                if len(current_word_parts) > 1:
                    full_word = "".join([part.replace("##", "") for part in current_word_parts])
                    split_word_freq[full_word] += 1
                current_word_parts = [token] if not token.startswith("##") else [token]
        if len(current_word_parts) > 1:
            full_word = "".join([part.replace("##", "") for part in current_word_parts])
            split_word_freq[full_word] += 1

    return split_word_freq

# Инициализация эмбеддингов новых слов
def initialize_new_embeddings(model, tokenizer, new_words):
    with torch.no_grad():
        current_embeddings = model.get_input_embeddings().weight.data

        for word in new_words:
            word_tokens = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            token_embeddings = current_embeddings[token_ids, :]
            new_embedding = token_embeddings.mean(dim=0)
            new_word_id = tokenizer.convert_tokens_to_ids(word)
            current_embeddings[new_word_id, :] = new_embedding
