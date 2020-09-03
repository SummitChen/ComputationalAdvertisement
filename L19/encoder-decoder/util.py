from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import jieba
import random
from vocabulary import *
import torch

import time
import math

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'

MAX_LENGTH = 10
SOS_token = 1
EOS_token = 2

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# turn a unicode string to plain ascii
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lowercase, trim and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs_test(file, reverse=False):
    print("Reading lines ...")

    lines = open('data/'+file, encoding='utf-8'). \
        read().strip().split('\n')

    # pairs = [[normalize_string(l.split('\t')[0]), l.split('\t')[1] ] for l in lines]

    pairs = [[normalize_string(l.split('\t')[0]).split(), list(jieba.cut(l.split('\t')[1])) ] for l in lines]

    for p in pairs[-10:]:
        print(p[0])
        print(p[1])
        print('-'*20)

    return pairs

def read_langs(file, reverse=False):
    print("Reading lines ...")

    lines = open('data/'+file, encoding='utf-8'). \
        read().strip().split('\n')

    samples = lines[0].split('\t')[:2]
    print("Data sample from {} to {}".format(samples[0], samples[1]))

    # pairs = [[normalize_string(l.split('\t')[0]), list(jieba.cut(l.split('\t')[1])) ] for l in lines]

    pairs = [[normalize_string(l.split('\t')[0]), l.split('\t')[1] ] for l in lines]

    print("Encoded sample from {} of {} ".format(pairs[0], len(pairs)))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Vocabulary("cn")
        output_lang = Vocabulary("eng")
    else:
        input_lang = Vocabulary("eng")
        output_lang = Vocabulary("cn")

    return input_lang, output_lang, pairs

# helper functions for sampling data
def filter_pair(p):
    # return len(p[0].split(' ')) < MAX_LENGTH and \
    #     p[0].startswith(eng_prefixes)

    return len(p[0].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# prepare data
def prepare_data(reverse = False, sample = True):
    input_lang, output_lang, pairs = read_langs('cmn.txt', reverse)
    
    print("Read {} sentence pairs".format(len(pairs)))
    print("Trimmed to {} sentence pairs" .format(len(pairs)))
    print("Counting words...")
 
    if sample:
        pairs = filter_pairs(pairs)

    print("Counted words:")
    print(input_lang.name, len(input_lang))
    print(output_lang.name, len(output_lang))
    
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs

# convert data to tensor
def indexes_from_sentence(lang, sentence):
    words = sentence.split(' ') if lang.name == 'eng' else list(jieba.cut(sentence))
    # return [lang.word2indexp[word] for word in words]
    return [lang(word) for word in words]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, output_tensor)

# helper function to print time

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds'%(m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# plot training results

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()

    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

    plt.show()

if __name__ == "__main__":
    # read_langs('cmn.txt')
    read_langs_test('cmn.txt')
    # s = "你好啊"
    # print("中文编码 {} to {}".format(s, normalize_string(s)))