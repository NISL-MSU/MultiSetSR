import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Adapted from https://github.com/SymposiumOrganization/ControllableNeuralSymbolicRegression/blob/main/src/ControllableNesymres/architectures/sym_encoder.py


class SymEncoder(nn.Module):
    def __init__(self, cfg, dummy_param):
        super().__init__()
        self.trg_pad_idx = cfg.trg_pad_idx

        self.tok_embedding = nn.Embedding(num_embeddings=cfg.num_tokens_condition,
                                           embedding_dim=cfg.embedding_dim_condition)

        self.trasf_enc = nn.TransformerEncoderLayer(d_model=cfg.embedding_dim_condition,
                                                    nhead=cfg.num_heads,
                                                    dim_feedforward=cfg.dec_pf_dim,
                                                    dropout=cfg.dropout,
                                                    )
        self.enc = nn.TransformerEncoder(self.trasf_enc, num_layers=cfg.cond_num_layers)

        self.cfg = cfg

        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=480, nhead=8),
            num_layers=6
        )
        self.dummy_param = dummy_param

    def set_train(self):
        self.enc.train()
        self.tok_embedding.train()

    def set_eval(self):
        self.enc.eval()
        self.tok_embedding.eval()

    def make_src_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask

    def forward(self, batch):
        # Pad batch of prior expressions
        max_length = max(len(sk) for sk in batch)  # Find the maximum sequence length
        # Pad the skeletons to match the maximum length
        padded_tensors = [torch.cat((sk, torch.zeros(max_length - len(sk)).cuda())) for sk in batch]
        # Combine the padded  prior expressions into a single tensor
        batch = pad_sequence(padded_tensors, batch_first=True).type(torch.int).cuda()

        symbolic_conditioning = batch.long().to(self.dummy_param.device)
        # mask = self.make_src_mask(symbolic_conditioning)

        # NOTE: This implementation does not use positional encoding
        encoder_input = self.tok_embedding(symbolic_conditioning)

        enc_embedding = self.enc(encoder_input.permute(1, 0, 2))  # src_key_padding_mask=mask.bool())

        enc_embedding = enc_embedding.permute(1, 0, 2)

        if torch.isnan(enc_embedding).any():
            breakpoint()

        return enc_embedding.mean(dim=1, keepdim=True)
