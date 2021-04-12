from torch import nn
import torch.nn.functional as F

class NNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, text_length, lstm_hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size,
                            num_layers=1, batch_first=True)
        self.max_pool = nn.MaxPool2d((text_length,1))
        self.fc1 = nn.Linear(lstm_hidden_size, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, text):
        a1 = self.embeddings(text)
        a2 = self.lstm(a1)[0]
        a3 = self.max_pool(a2).squeeze(1)
        a4 = F.relu(self.fc1(a3))
        a5 = self.fc2(a4)
        return a5