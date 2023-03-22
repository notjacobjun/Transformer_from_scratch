import torch
from torch import nn
from models.networks.decoder import Decoder

from models.networks.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6, forward_expansion=4, heads=8, dropout_prob=0, device='cuda', max_length=100) -> None:
        super(Transformer, self).__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(heads, max_length, device, embed_size, num_layers,
                               dropout_prob, src_vocab_size, forward_expansion=forward_expansion)

        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers,
                               heads, forward_expansion, dropout_prob, device, max_length)

    def make_src_mask(self, src):
        # If we are at a src_pad_idx then we set the value to 0, otherwise 1
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # parse the dimensions from the target matrix
        N, trg_len = trg.shape
        # convert all the entries in the lower triangular half to one (using lower triangular matrix mask)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.LongTensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.LongTensor([[1, 7, 4, 3, 5, 9, 2, 0], [
        1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=9, heads=1, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
