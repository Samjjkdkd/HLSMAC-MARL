import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNNewAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNewAgent, self).__init__()
        self.args = args

        # 更深的网络结构：增加了一个额外的全连接层
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        x = F.relu(self.fc3(h))
        q = self.fc4(x)
        return q, h
