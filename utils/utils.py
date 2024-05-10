import torch


class Utils:
    def __init__(self, lang=None, sentence=None, input_lang=None, output_lang=None, pair=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EOS_token = 1
        self.lang = lang
        self.sentence = sentence
        self.pair = pair
        self.input_lang = input_lang
        self.output_lang = output_lang

    def indexesFromSentence(self):
        return [self.lang.word2index[word] for word in self.sentence.split(' ')]

    def tensorFromSentence(self):
        indexes = self.indexesFromSentence()
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(1, -1)

    def tensorsFromPair(self):
        input_tensor = self.tensorFromSentence()
        target_tensor = self.tensorFromSentence()
        return (input_tensor, target_tensor)
