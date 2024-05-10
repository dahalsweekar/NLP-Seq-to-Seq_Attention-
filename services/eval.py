import torch
from utils.utils import Utils

class Eval:
    def __init__(self, encoder, decoder, sentence, input_lang, output_lang, number_of_samples):
        self.encoder = encoder
        self.decoder = decoder
        self.sentence = sentence
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.EOS_token = 1
        self.n = number_of_samples

    def evaluate(self):
        with torch.no_grad():
            input_tensor = Utils(self.input_lang, self.sentence).tensorFromSentence()

            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(self.output_lang.index2word[idx.item()])
        return decoded_words, decoder_attn
