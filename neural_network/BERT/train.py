import torch

from transformers import AddedToken
from tqdm.auto import tqdm, trange

from models import classification_bert
from utils import utils, tokens, dataset, evaluation
import config


print(config.device)

model, tokenizer = classification_bert.get_model_and_tokenizer(config.BERT_MODEL_NAME, config.LABELS)

# Загрузка данных
train_texts, train_labels = utils.load_data('../train_data.csv')
test_texts, test_labels = utils.load_data('../test_data.csv')



# Находим разбитые на части слова
split_word_freq = tokens.find_split_words(train_texts, tokenizer)

# Топ-10 самых частых слов
top_10_words = sorted(split_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 most common split words:")
for word, freq in top_10_words:
    print(f"{word}: {freq}")

# Выбираем только те слова, которые встречаются больше порога
most_common_split_words = [word for word, freq in split_word_freq.items() if freq > 30]

# Добавление самых популярных слов в словарь токенизатора с атрибутом single_word
new_tokens = [AddedToken(word, single_word=True) for word in most_common_split_words]
num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
print(f"Added {num_added_tokens} tokens")


# Инициализация эмбеддингов для новых слов
model.resize_token_embeddings(len(tokenizer))
tokens.initialize_new_embeddings(model, tokenizer, most_common_split_words)



# Загрузка данных
train_dataset = dataset.TextDataset(train_texts, train_labels, tokenizer, model)
test_dataset = dataset.TextDataset(test_texts, test_labels, tokenizer, model)

batch_size = 16
train_dataloader, dev_dataloader = dataset.get_dataloader(batch_size, train_dataset, test_dataset)


utils.cleanup()


# zero-shot классификация
f1 = evaluation.calculate_f1(model, dev_dataloader)
print(f'\n[epoch 0] val f1: {f1:.4f}\n\n')


# Обучение модели
optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .5 ** epoch)

for epoch in trange(4):
    model.train()
    utils.cleanup()

    print('LR:', scheduler.get_last_lr())
    tq = tqdm(train_dataloader, desc='Training', leave=False)
    for i, batch in enumerate(tq):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tq.set_postfix(loss=loss.item())

    val_loss = evaluation.calculate_validation_loss(model, dev_dataloader)
    scheduler.step()

    f1 = evaluation.calculate_f1(model, dev_dataloader)
    print(f'\n[epoch {epoch + 1}] val f1: {f1:.4f}, val loss: {val_loss:.4f}\n\n')

    model.save_pretrained(config.MODEL_PATH)
    tokenizer.save_pretrained(config.MODEL_PATH)
    print(f'\n[epoch {epoch + 1}] model saved')

utils.cleanup()
