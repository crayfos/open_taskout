import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=True, dropout=0.5):
        super(AttentionRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        attn_weights = F.softmax(self.attention(output), dim=1)
        weighted_output = torch.sum(output * attn_weights, dim=1)
        weighted_output = self.dropout(weighted_output)
        logits = self.fc(weighted_output)
        return logits