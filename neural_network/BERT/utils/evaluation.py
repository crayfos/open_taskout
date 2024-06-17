import torch
import numpy as np
from sklearn.metrics import f1_score
from web.neural_network.BERT2 import config


def evaluate_model(model, dataloader, tokenizer=None):
    model.eval()
    true_labels = []
    predictions = []
    texts = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)

        true_labels.append(labels.cpu().numpy())

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions.append(probs.cpu().numpy())


        if tokenizer is not None:
            texts.extend([tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids])

    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)
    return true_labels, predictions, texts



def calculate_f1(model, dataloader):
    model.eval()
    facts, preds, _ = evaluate_model(model, dataloader)

    preds = np.argmax(preds, axis=1)
    facts = np.argmax(facts, axis=1)

    f1 = f1_score(facts, preds, average='weighted')
    return f1



def calculate_validation_loss(model, validation_dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(validation_dataloader)