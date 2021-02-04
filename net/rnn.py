import torch.nn as nn
import torch.nn.functional as F


class rnn(nn.Module):
    def __init__(self, input_size, output_size):
        super(rnn, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=False,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=False,
                            bidirectional=True)
        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.2)
        self.Seq_1 = nn.Sequential(nn.ReLU(), self.rnn1)
        self.Seq_2 = nn.Sequential(nn.ReLU(), self.rnn2)
        self.linear_1 = nn.Sequential(nn.Linear(256, 64))
        self.linear_2 = nn.Sequential(nn.Linear(64, output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output, (h_n, c_n) = self.Seq_1(input)
        output = self.drop_1(output)
        output, (h_n, c_n) = self.Seq_2(output)
        output = self.drop_2(output)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.sigmoid(output[-1])
        return output

class rnn_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(rnn_2, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=False,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=64,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=False,
                            bidirectional=False)
        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.2)
        self.Seq_1 = nn.Sequential(nn.ReLU(), self.rnn1)
        self.Seq_2 = nn.Sequential(nn.ReLU(), self.rnn2)
        self.linear_1 = nn.Sequential(nn.Linear(128, 128))
        self.linear_2 = nn.Sequential(nn.Linear(128, 64))
        self.linear_3 = nn.Sequential(nn.Linear(64, output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output, (h_n, c_n) = self.Seq_1(input)
        output = self.drop_1(output)
        output, (h_n, c_n) = self.Seq_2(output)
        output = self.drop_2(output)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = F.relu(output[-1])
        return output
