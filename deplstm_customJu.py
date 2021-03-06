import torch
import torch.nn as nn
import settings


class LockedDropoutForAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - dropout) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class HistoryAttention(nn.Module):
    # def __init__(self, hidden_size, att_hidden_size, max_history_size=15, p=0.5):
    def __init__(self, input_size, hidden_size, att_hidden_size, max_history_size=15, p=0.5):  # Ju
        super(HistoryAttention, self).__init__()
        # self.history_net = nn.Linear(hidden_size, att_hidden_size)
        # self.hidden_net = nn.Linear(hidden_size, att_hidden_size)
        self.history_net = nn.Linear(hidden_size, 1)
        # self.hidden_net = nn.Linear(hidden_size, 1)
        self.hidden_net = nn.Linear(input_size, 1)  # Ju

        # self.attention_net = nn.Conv1d(att_hidden_size, 1, 1, 1)
        # self.attention_net = nn.Conv1d(2, 1, 1, 1)

        # self.projection_net = nn.Linear(hidden_size, hidden_size)

        # self.locked_dropout = LockedDropoutForAttention()
        self.dropout = nn.Dropout(p=0.4)

        self.max_history_size = max_history_size
        self.p = p

        self.history_embs = []
        self.history = []
        self.dropmask = None

        self.itr = 0

    def forward(self, current, previous):  # both in B x hidden_size

        r"""For debugging"""
        # if torch.max(current.view(-1)) > 10 or torch.max(previous.view(-1)) > 10:
        #     print('trap')
        # print(previous.view(-1)[(previous.view(-1) > 10).nonzero()[0][0].item()].item())
        # print(current.view(-1)[(current.view(-1) > 10).nonzero()[0][0].item()].item())
        r"""==============="""

        if len(self.history) == len(self.history_embs):
            self.history.append(previous)  # list of B x hidden_size tensors
            previous_emb = self.history_net(previous)  # B x att_hidden_size
            self.history_embs.append(self.dropout(previous_emb))

            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]
                self.history_embs = self.history_embs[-self.max_history_size:]
        else:
            self.history.append(previous)
            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]

            self.history_embs = [self.history_net(self.dropout(h)) for h in self.history]

        current = self.dropout(current)
        current_emb = self.hidden_net(current)  # B x att_hidden_size

        history_embs = torch.stack(self.history_embs, dim=2)  # B x att_hidden_size x history_length

        r"""Two layer perceptron: concat (prev context, current word) -> att probs"""
        # if len(self.history) > 1:
        #     history_embs_reshaped = torch.stack(self.history_embs).squeeze().t().unsqueeze(1)  # Ju
        # else:
        #     history_embs_reshaped = torch.stack(self.history_embs).squeeze(0).unsqueeze(2)
        # current_embs_reshaped = current_emb.unsqueeze(0).squeeze().unsqueeze(1).unsqueeze(2).expand_as(
        #     history_embs_reshaped)  # Ju
        # att_scores = self.attention_net(torch.cat((history_embs_reshaped, current_embs_reshaped), dim=1))  # Ju
        # att_probs = torch.softmax(att_scores, dim=2)  # Ju
        r"""======================================================================="""

        # B x att_hidden_size x history_length
        # hidden_embs = self.locked_dropout(torch.tanh(history_embs + current_emb.unsqueeze(2).expand_as(history_embs)),
        #                                   dropout=self.p)
        # hidden_embs = torch.tanh(history_embs + current_emb.unsqueeze(2).expand_as(history_embs))

        # att_scores = self.attention_net(hidden_embs)  # B x 1 x history_length
        att_probs = torch.softmax(history_embs + current_emb.unsqueeze(2).expand_as(history_embs), dim=2)
        # att_probs = torch.softmax(att_scores, dim=2)  # B x 1 x history_length

        history = torch.stack(self.history, dim=2)  # B x hidden_size x history_length
        attended_history = history * att_probs.expand_as(history)  # B x hidden_size x history_length
        # attended_history = self.dropout(attended_history.sum(dim=2))  # B x hidden_size
        attended_history = attended_history.sum(dim=2)  # B x hidden_size

        # return torch.tanh(self.projection_net(attended_history))
        return attended_history

    def reset_history(self):
        self.history = []
        self.history_embs = []

    def detach_history(self):
        history = []
        for h in self.history:
            history.append(h.detach())
        self.history = history
        self.history_embs = []
        self.dropmask = None


class DepLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False, att_hidden_size=None, max_attention_size=None):
        super(DepLSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        att_hidden_size = att_hidden_size if att_hidden_size is not None else hidden_size
        max_attention_size = max_attention_size if max_attention_size is not None else 15
        # self.attention = HistoryAttention(hidden_size, att_hidden_size, max_attention_size)
        self.attention = HistoryAttention(input_size, hidden_size, att_hidden_size, max_attention_size)  # Ju

        # self.gates = nn.Linear(hidden_size+att_hidden_size, hidden_size+att_hidden_size, max_attention_size)
        self.gates = nn.Linear(hidden_size+hidden_size, hidden_size)

    def forward(self, inputs, hx=None):
        if hx is None:
            prev = inputs.new_zeros(inputs.size(1), self.hidden_size, requires_grad=False)
        else:
            prev = hx[0][0]

        self.attention.detach_history()
        outputs = []
        for input in inputs:
            _, new_hx = super(DepLSTM, self).forward(input.unsqueeze(0), hx)
            # if not self.training:
            #     new_hx = (new_hx[0].detach(), new_hx[1].detach())

            # attended_feat = self.attention(new_hx[0][0], prev)
            attended_feat = self.attention(input, prev)  # Ju
            gates = torch.sigmoid(self.gates(torch.cat([attended_feat, new_hx[0][0]], dim=1)))
            # new_h = (attended_feat * gates[:, :attended_feat.size(1)]).unsqueeze(0) \
            #         + (new_hx[0][0] * gates[:, -new_hx[0][0].size(1):]).unsqueeze(0)
            new_h = (attended_feat * gates).unsqueeze(0) + (new_hx[0][0] * (1-gates)).unsqueeze(0)
            outputs.append(new_h)
            hx = (new_h, new_hx[1])
            prev = new_h[0]

        return torch.cat(outputs, dim=0), hx


if __name__ == '__main__':
    # lstm = nn.LSTM(3, 3)
    deplstm = DepLSTM(3, 3)

    inputs = [torch.randn(1, 3) for _ in range(2)]  # make a sequence of length 5
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)

    # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
    # out, hidden = lstm(inputs, hidden)
    # print(out)
    # print(hidden)

    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
    out, hidden = deplstm(inputs, hidden)
    print(out)
    print(hidden)




