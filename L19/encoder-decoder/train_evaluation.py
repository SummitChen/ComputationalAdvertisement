from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu

from rnn import *
from util import *
from vocabulary import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'

MAX_LENGTH = 10
SOS_token = 1
EOS_token = 2
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
            criterion, max_length = MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        # # debug

        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            
            # # debug
            if isinstance(decoder, AttnDecoderRNN):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            # # debug
            if isinstance(decoder, AttnDecoderRNN):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, pairs, input_lang, output_lang,
        print_every = 1000, plot_every = 100, learning_rate = 0.01):
    
    start = time.time()

    plot_losses = []

    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()

    # print("input format {} - {}".format(training_pairs[0][0].size(),training_pairs[0][0]))

    for iter in range(n_iters):

        c_iter = iter + 1

        training_pair = training_pairs[iter]
        input_tensor = training_pair[0]
        output_tensor = training_pair[1]

        # print("Iter - {} input size {} target size {}".format(iter, input_tensor.size(), output_tensor.size()))

        loss = train(input_tensor, output_tensor, encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if c_iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d, %d%%) %.4f' % (time_since(start, c_iter / n_iters),
                                          c_iter, c_iter / n_iters * 100, print_loss_avg))

        if c_iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    print("plot losses {}".format(len(plot_losses)))
    show_plot(plot_losses)

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder_hidden = encoder.init_hidden()
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, hidden_size, device = device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):

            if isinstance(decoder, AttnDecoderRNN):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                
                decoder_attentions[di] = decoder_attention.data
            
            else:

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            
            _, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        if isinstance(decoder, AttnDecoderRNN):
            return decoded_words, decoder_attentions[:di + 1]
        else:
            return decoded_words

def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n = 10):

    bleu_scores = []

    for i in range(10):
        pair = random.choice(pairs)
        print('> {}'.format(pair[0]))
        print('= {}'.format(pair[1]))
        
        if isinstance(decoder, AttnDecoderRNN):
            output_words, _ = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
        else:
            output_words = evaluate(encoder, decoder, input_lang, output_lang, pair[0])

        output_sentence = ''.join(output_words)

        reference_sentence = [list(pair[1].strip())]
        score = sentence_bleu(reference_sentence, output_words)
        bleu_scores.append(score)
        print('< {}'.format(output_sentence))

        print(' ')
    
    print('Avg BLEU score is {}'.format(np.sum(bleu_scores)/len(bleu_scores)))



if __name__ == "__main__":
    hidden_size = 256
    # input_lang, output_lang, pairs = read_langs('cmn.txt')

    # training 
    encoder = None
    decoder = None

    load_pretrained = True

    input_lang, output_lang, pairs = prepare_data()

    encoder = EncoderRNN(len(input_lang), hidden_size).to(device)
    # decoder = DecoderRNN(hidden_size, len(output_lang)).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(output_lang), dropout_p=0.1).to(device)

    if load_pretrained:
        encoder = torch.load('models/encoder.pt')
        decoder = torch.load('models/attndecoder.pt')

    trainIters(encoder, decoder, 80000, pairs, input_lang, output_lang, print_every=5000)

    torch.save(encoder, 'models/encoder.pt')
    torch.save(decoder, 'models/attndecoder.pt')

    if encoder == None or decoder == None:
        encoder = torch.load('models/encoder.pt')
        decoder = torch.load('models/attndecoder.pt')

    evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs)
