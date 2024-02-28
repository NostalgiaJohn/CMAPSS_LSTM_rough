from macro import *
import torch
import torch.nn as nn
# from torch.autograd import Variable

class LSTM1(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
        # super(LSTM1, self).__init__()
        # nn.Module.__init__(self)
        super().__init__()

        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        # self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # this option will change input data arangement from
                              # (sequence_length, batch_size, input_size/feature) to
                              # (batch_size, sequence_length, input_size/feature)
            dropout=0.1
        )
        # self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        # self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        # self.fc = nn.Linear(8, 1)  # fully connected last layer
        #
        # self.dropout = nn.Dropout(p=0.1)
        # self.relu = nn.ReLU()

        self.relu_fc_dropout_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features  = hidden_size,
                out_features = 16
            ),

            nn.ReLU(),
            nn.Linear(
                in_features  = 16,
                out_features =8
            ),

            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features  = 8,
                out_features = 1
            )
        )

    def forward(self, x):
        """

        :param x: input features
        :return: prediction results
        # """
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=DEVICE))  # hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=DEVICE))  # internal state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)  # internal state
        """
        Here torch.rand or torch.randn is not recommended for h_0's and c_0's
        initialization, cuz it may cause gradient vanishing and explosing problems.
        """

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn_o = torch.tensor(hn.to(CPU).detach().numpy()[-1, :, :]).to(DEVICE)
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = torch.tensor(hn.to(CPU).detach().numpy()[ 1, :, :]).to(DEVICE)
        hn_1 = hn_1.view(-1, self.hidden_size)

        # out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        # out = self.relu(self.fc_2(out))
        # out = self.dropout(out)
        # out = self.fc(out)
        out = self.relu_fc_dropout_stack(hn_o + hn_1)
        return out
