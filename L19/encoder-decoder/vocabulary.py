import jieba
# Vacabulary List
class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        # self.idx2word = {0: "<SOS>", 1: "<EOS>", -1: "<unk>"}
        # self.idx2word = {0: "<SOS>", 1: "<EOS>"}
        self.idx2word = {1: "<SOS>", 2: "<EOS>", 0: "<unk>"}
        # self.idx = 2
        self.idx = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def add_sentence(self, sentence):
        words = list(jieba.cut(sentence)) if self.name == "cn" else sentence.split()
        for word in words:
            self.add_word(word)

    def __call__(self, word):
        if word not in self.word2idx:
            # print("Did not find word {}".format(word))
            # return -1
            return 0
        return self.word2idx[word]
    
    def __len__(self):
        return self.idx
