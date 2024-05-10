import time
import math
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils.plotter import Plot


class Train:
    def __init__(self, dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
                 print_every=100, plot_every=100):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.plot_every = plot_every

    def train_epoch(self, encoder_optimizer,
                    decoder_optimizer, criterion):
        total_loss = 0
        i = 0
        for data in tqdm(self.dataloader, desc=f'Epoch {i + 1}/ {self.n_epochs}'):
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            i += 1

        return total_loss / len(self.dataloader)

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def train(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        criterion = nn.NLLLoss()

        for epoch in range(1, self.n_epochs + 1):
            loss = self.train_epoch(encoder_optimizer,
                                    decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, epoch / self.n_epochs),
                                             epoch, epoch / self.n_epochs * 100, print_loss_avg))

            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        Plot(plot_losses).showPlot()
