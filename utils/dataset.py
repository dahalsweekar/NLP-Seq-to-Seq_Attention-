from __future__ import unicode_literals, print_function, division
from utils.lang import Lang
import unicodedata
import re


class Dataset:
    def __init__(self, lang1, lang2, reverse, MAX_LENGTH, data_path):
        self.MAX_LENGTH = MAX_LENGTH
        self.eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        self.lang1 = lang1
        self.lang2 = lang2
        self.reverse = reverse
        self.data_path = data_path

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()

    def readLangs(self):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open(f'{self.data_path}/%s-%s.txt' % (self.lang1, self.lang2), encoding='utf-8'). \
            read().strip().split('\n')

        # Split every line into pairs and normalize
        if self.lang2 == "npi":
            new_pairs = []
            pairs = [[s for s in l.split('\t')] for l in lines]
            for pair in pairs:
                if len(pair[:-1]) == 2:
                    new_pairs.append(pair[:-1])
            pairs = new_pairs
        else:
            pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)

        return input_lang, output_lang, pairs

    def filterPair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and \
            len(p[1].split(' ')) < self.MAX_LENGTH and \
            p[1].startswith(self.eng_prefixes)

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def prepareData(self):
        input_lang, output_lang, pairs = self.readLangs()
        print("Read %s sentence pairs" % len(pairs))
        if self.lang2 != "npi":
            pairs = self.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
