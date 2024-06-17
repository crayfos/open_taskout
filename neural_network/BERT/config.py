import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ['разработка', 'дизайн', 'контент', 'маркетинг', 'бизнес', 'другое']

BERT_MODEL_NAME = "cointegrated/rubert-tiny2"
MODEL_PATH = "./freelance_bert"

TRAIN_DATA_PATH = '../train_data.csv'
TEST_DATA_PATH = '../test_data.csv'
