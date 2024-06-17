# Модель на основе стандартной обёртки BertForSequenceClassification
from transformers import BertForSequenceClassification, AutoTokenizer, AddedToken
from web.neural_network.BERT2 import config


def get_model_and_tokenizer(model_name, labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        problem_type='multi_label_classification'
    ).to(config.device)

    model.config.label2id = {label: i for i, label in enumerate(labels)}
    model.config.id2label = {i: label for i, label in enumerate(labels)}

    return model, tokenizer