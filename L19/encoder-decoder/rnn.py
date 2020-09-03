import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'
MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        print("Encoder Init hidden_size: {} input_size: {}".format(hidden_size, input_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        # print("input shape{}, embedding shape {} embeded shape {}".format(input.size(), self.embedding(input).size(), embedded.size()))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        print("Decoder Init hidden_size: {} ouput_size: {}".format(hidden_size, output_size))
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.debug = False

    def forward(self, input, hidden):
        embeded = self.embedding(input).view(1, 1, -1)
        relued = F.relu(embeded)
        output_raw, hidden = self.gru(relued, hidden)
        output = self.softmax(self.out(output_raw[0]))

        if self.debug == True:
            print(f"input: {input.size()}, embeded: {embeded.size()}, relued: {relued.size()} output_raw: {output_raw.size()}")
            print(f"hidden: {hidden.size()}, output: {output.size()}")

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device= device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.debug = False

    def forward(self, input, hidden, encoder_outputs):
        embeded = self.embedding(input).view(1, 1, -1)
        embeded = self.dropout(embeded)

        if self.debug == True:
            print(f"input: {input.size()}, hidden: {hidden.size()} encoder_outputs: {encoder_outputs.size()}")

        attn_weights = F.softmax(
            self.attn(torch.cat((embeded[0], hidden[0]), 1)), dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embeded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        if self.debug == True:
            print(f"hidden: {hidden.size()} attn_weights: {attn_weights.size()} attn_applied: {attn_applied.size()} output: {output.size()}")
            self.debug = False

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)

        

