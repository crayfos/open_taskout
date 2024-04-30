import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, sequence_length, embed_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, sequence_length]
        x = [F.relu(conv(x)) for conv in self.convs]  # Apply Convolution + ReLU for each filter size
        x = [F.max_pool1d(line, line.shape[2]).squeeze(2) for line in x]  # Apply Max Pooling for each conv. output
        x = torch.cat(x, 1)  # Concatenate the conv. outputs
        x = self.dropout(x)  # Apply Dropout
        x = self.fc(x)  # Final Fully Connected Layer
        return x
