import torch
from torch import nn
import torch.nn.functional as F
from .set_transformer import PMA
from .set_encoder import SetEncoder
from .beam_search import BeamHypotheses


class Model(nn.Module):
    def __init__(self, cfg, cfg_inference, word2id, loss=None):
        super(Model, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.enc = SetEncoder(cfg)
        self.trg_pad_idx = cfg.trg_pad_idx
        self.cfg = cfg
        self.tok_embedding = nn.Embedding(self.cfg.output_dim, self.cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(self.cfg.length_eq, self.cfg.dim_hidden)
        if cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                self.cfg.length_eq, self.cfg.dim_hidden, out=self.pos_embedding.weight
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.dim_hidden,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.dec_pf_dim,
            dropout=self.cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        self.aggregator = PMA(self.cfg.dim_hidden, self.cfg.num_heads, 1)
        self.cfg_inference = cfg_inference
        self.word2id = word2id
        self.dropout = nn.Dropout(cfg.dropout)
        self.eq = None
        self.loss = loss

        self.dummy_param = nn.Parameter(torch.empty(0))

    def set_train(self):
        self.enc.train()
        self.decoder_transfomer.train()
        self.fc_out.train()
        self.dropout.train()
        self.tok_embedding.train()
        self.pos_embedding.train()

    def set_eval(self):
        self.enc.eval()
        self.decoder_transfomer.eval()
        self.fc_out.eval()
        self.dropout.eval()
        self.tok_embedding.eval()
        self.pos_embedding.eval()

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def forward(self, batch, skeleton):
        # Separate target skeleton, mask it, and create embeddings
        trg = skeleton
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        pos = self.pos_embedding(
            torch.arange(0, int(skeleton.shape[1] - 1))
            .unsqueeze(0)
            .repeat(skeleton.shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)

        # Separate input sets and apply the encoder layer to each one
        n_sets = batch.shape[-1]
        z_sets = torch.Tensor().to(self.dummy_param.device)
        for i_set in range(n_sets):
            enc_src = self.enc(batch[:, :, :, i_set].to(self.dummy_param.device))
            assert not torch.isnan(enc_src).any()
            z_sets = torch.cat((z_sets, enc_src), dim=1)

        # Merge outputs from all sets
        enc_output = self.aggregator(z_sets)
        # enc_output = torch.sum(z_sets, dim=0, keepdim=True)

        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_output.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        )

        output = self.fc_out(output)

        if torch.cuda.device_count() > 1:
            # Calcualte loss
            L1 = torch.zeros(1).to(self.dummy_param.device)
            output2 = torch.clone(output)
            # print(output.shape)
            for bi in range(output.shape[1]):
                out = output[:, bi, :].contiguous().view(-1, output.shape[-1])
                tokenized = skeleton[bi, :][1:].contiguous().view(-1)
                L1s = self.loss(out, tokenized.long())
                L1 += L1s
            return output2, z_sets, L1
        else:
            return output, z_sets

    def validation_step(self, batch, skeleton):
        """Perform validation"""
        with torch.no_grad():
            #############################################################
            # ENCODER
            #############################################################
            # Separate input sets and apply the encoder layer to each one
            n_sets = batch.shape[-1]
            z_sets = torch.Tensor().to(self.dummy_param.device)
            for i_set in range(n_sets):
                enc_src = self.enc(batch[:, :, :, i_set])
                assert not torch.isnan(enc_src).any()
                z_sets = torch.cat((z_sets, enc_src), dim=1)

            # Merge outputs from all sets
            enc_output = self.aggregator(z_sets)
            # enc_output = torch.sum(z_sets, dim=0, keepdim=True)

            #############################################################
            # DECODER
            #############################################################
            # Separate target skeleton, mask it, and create embeddings
            trg = skeleton
            trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
            pos = self.pos_embedding(
                torch.arange(0, int(skeleton.shape[1] - 1))
                .unsqueeze(0)
                .repeat(skeleton.shape[0], 1)
                .type_as(trg)
            )
            te = self.tok_embedding(trg[:, :-1])
            trg_ = self.dropout(te + pos)
            output = self.decoder_transfomer(
                trg_.permute(1, 0, 2),
                enc_output.permute(1, 0, 2),
                trg_mask2.bool(),
                tgt_key_padding_mask=trg_mask1.bool()
            )
            outputs = self.fc_out(output)

            # # Perform autoregressive generation
            # outputs = []
            # seqs = []
            # for b in range(batch.size(0)):
            #     tgt = torch.tensor([1]).to(self.dummy_param.device)[None, :]
            #     seq = []
            #     scores = torch.Tensor().to(self.dummy_param.device)
            #     for i in range(self.cfg.length_eq):
            #         pos = self.pos_embedding(
            #             torch.arange(0, tgt.shape[1])
            #             .unsqueeze(0)
            #             .type_as(tgt)
            #         )
            #         te = self.tok_embedding(tgt)
            #         tgt_ = self.dropout(te + pos)
            #         # Forward pass through the decoder using tgt and memory
            #         output = self.decoder_transfomer(tgt_.permute(1, 0, 2),
            #                                          memory=enc_output[b:b+1, :, :].permute(1, 0, 2))
            #         output = self.fc_out(output)
            #         scores = torch.cat((scores, output[-1, :, :]), dim=0)
            #
            #         # Sample the next token from the probability distribution
            #         next_token = torch.argmax(output[-1, :, :], dim=-1)
            #         seq.append(int(next_token.cpu()))
            #
            #         if i < self.cfg.length_eq - 1:
            #             # Append the next token to tgt for the next step
            #             tgt = torch.cat([tgt, next_token[None, :]], dim=-1)
            #         if next_token == 2:
            #             break
            #     outputs.append(scores)
            #     seqs.append(seq)
        return outputs  # , seqs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def inference(self, batch):
        """Perform inference using beam search"""
        with torch.no_grad():
            #############################################################
            # ENCODER
            #############################################################
            # Separate input sets and apply the encoder layer to each one
            n_sets = batch.shape[-1]
            z_sets = torch.Tensor().to(self.dummy_param.device)
            for i_set in range(n_sets):
                enc_src = self.enc(batch[:, :, :, i_set])
                assert not torch.isnan(enc_src).any()
                z_sets = torch.cat((z_sets, enc_src), dim=1)

            # Merge outputs from all sets
            enc_output = self.aggregator(z_sets)

            src_enc = enc_output
            shape_enc_src = (self.cfg_inference.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand(
                (1, self.cfg_inference.beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)
            # print("Memory footprint of the encoder: {}GB \n".
            #       format(enc_src.element_size() * enc_src.nelement() / 10 ** 9))

            #############################################################
            # DECODER AND BEAM SEARCH
            #############################################################
            generated = torch.zeros([self.cfg_inference.beam_size, self.cfg.length_eq], dtype=torch.long,
                                    device=self.dummy_param.device, )
            generated[:, 0] = 1  # Initialize with SOS symbol
            cache = {"slen": 0}
            generated_hyps = BeamHypotheses(self.cfg_inference.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False
            # Beam Scores
            beam_scores = torch.zeros(self.cfg_inference.beam_size, device=self.dummy_param.device, dtype=torch.long)
            beam_scores[1:] = -1e9

            cur_len = torch.tensor(1, device=self.dummy_param.device, dtype=torch.int64)
            # Repeat until maximum length is reached
            while cur_len < self.cfg.length_eq:
                # Create masks
                generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])
                # Embeddings
                pos = self.pos_embedding(
                    torch.arange(0, int(cur_len))
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)
                # Feed decoder
                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                # Apply softmax to obtain probability distributions
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)

                n_words = scores.shape[-1]
                # Select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)
                _scores = _scores.view(self.cfg_inference.beam_size * n_words)

                next_scores, next_words = torch.topk(_scores, 2 * self.cfg_inference.beam_size, dim=0, largest=True,
                                                     sorted=True)
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # Next words for this sentence
                for idx, value in zip(next_words, next_scores):
                    # Get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    # End of sentence, or next word
                    if word_id == self.word2id["F"] or cur_len + 1 == self.cfg.length_eq:
                        generated_hyps.add(generated[beam_id, :cur_len, ].clone().cpu(), value.item(), )
                    else:
                        next_sent_beam.append((value, word_id, beam_id))

                    # The beam for next step is full
                    if len(next_sent_beam) == self.cfg_inference.beam_size:
                        break

                # Update next beam content
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.trg_pad_idx, 0)] * self.cfg_inference.beam_size  # pad the batch

                beam_scores = torch.tensor([x[0] for x in next_sent_beam], device=self.dummy_param.device)
                beam_words = torch.tensor([x[1] for x in next_sent_beam], device=self.dummy_param.device)
                beam_idx = torch.tensor([x[2] for x in next_sent_beam], device=self.dummy_param.device)
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # Update current length
                cur_len += torch.tensor(1, device=self.dummy_param.device, dtype=torch.int64)

            return sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True)

    def get_equation(self, ):
        return self.eq


if __name__ == "__main__":
    print("model")
