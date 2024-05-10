import numpy as np
import torch
from utils.dataset import Dataset
from utils.utils import Utils
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class DL:
    def __init__(self, lang1, lang2, reverse, MAX_LENGTH, batch_size, data_path):
        self.batch_size = batch_size
        self.lang1 = lang1
        self.lang2 = lang2
        self.reverse = reverse
        self.EOS_token = 1
        self.MAX_LENGTH = MAX_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path

    def get_dataloader(self):
        input_lang, output_lang, pairs = Dataset(self.lang1, self.lang2, self.reverse, self.MAX_LENGTH,
                                                 self.data_path).prepareData()

        n = len(pairs)
        input_ids = np.zeros((n, self.MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, self.MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = Utils(input_lang, inp).indexesFromSentence()
            tgt_ids = Utils(output_lang, tgt).indexesFromSentence()
            inp_ids.append(self.EOS_token)
            tgt_ids.append(self.EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(input_ids).to(self.device),
                                   torch.LongTensor(target_ids).to(self.device))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        return input_lang, output_lang, train_dataloader
