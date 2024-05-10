from __future__ import unicode_literals, print_function, division
import random

import torch
from utils.dataset import Dataset
from services.train import Train
from models.RNNmodel import EncoderRNN
from models.AttentionModel import AttnDecoderRNN
from services.eval import Eval
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

from utils.dataloader import DL


def main():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--evaluate', help='Evaluate', default='True', action='store_true')
    parser.add_argument('--lang1', help='Primary Language', default='eng')
    parser.add_argument('--lang2', help='Secondary Language', default='npi')
    parser.add_argument('--reverse', help='Reverse?', default=False, action='store_true')
    parser.add_argument('--max_length', type=int, help='Filter Length', default=50)
    parser.add_argument('--epoch', type=int, help='number of epoches', default=50)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--data_path', help='root path to dataset',
                        default='./data')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    Run(args['lang1'], args['lang2'], args['reverse'], args['max_length'],
        args['epoch'], args['batch_size'], args['data_path'], args['evaluate']).invoke()


class Run:
    def __init__(self, lang1, lang2, reverse, max_length, epoch, batch_size, data_path, evaluate):
        self.lang1 = lang1
        self.lang2 = lang2
        self.reverse = reverse
        self.max_length = max_length
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_path = data_path
        self.evaluate = evaluate

    def invoke(self):
        input_lang, output_lang, pairs = Dataset(self.lang1, self.lang2, self.reverse, self.max_length,
                                                 data_path=self.data_path).prepareData()
        print(random.choice(pairs))

        input_lang, output_lang, train_dataloader = DL(self.lang1, self.lang2, self.reverse, self.max_length,
                                                       self.batch_size, self.data_path).get_dataloader()

        hidden_size = 128

        #
        # input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
        #
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH=self.max_length).to(device)
        #
        print('Training...')
        Train(dataloader=train_dataloader, encoder=encoder, decoder=decoder, n_epochs=self.epoch, print_every=5,
              plot_every=5).train()

        if self.evaluate:
            print('Starting Evaluation...')
            encoder.eval()
            decoder.eval()
            self.evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, number_of_samples=10)

    def evaluateRandomly(self, encoder, decoder, pairs, input_lang, output_lang, number_of_samples=10):
        for i in range(number_of_samples):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, _ = Eval(encoder, decoder, pair[0], input_lang, output_lang, number_of_samples).evaluate()
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')


if __name__ == '__main__':
    main()
