import torch
import torch.nn as nn
from .set_transformer import ISAB, PMA, SAB


class SetEncoder(nn.Module):
    def __init__(self, cfg):
        super(SetEncoder, self).__init__()
        self.linear = cfg.linear
        self.bit16 = cfg.bit16
        self.norm = cfg.norm
        assert cfg.linear != cfg.bit16, "one and only one between linear and bit16 must be true at the same time"
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))

        self.activation = cfg.activation
        self.input_normalization = cfg.input_normalization
        if cfg.linear:
            self.linearl = nn.Linear(cfg.dim_input, 16 * cfg.dim_input)

        # Encoder structure (stack of SABs) IMPORTANT: Using SABs instead of ISABs (SABs require fewer parameters)
        self.selfatt = nn.ModuleList()
        self.selfatt1 = ISAB(16 * cfg.dim_input, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln)
        for i in range(cfg.n_l_enc):
            self.selfatt.append(ISAB(cfg.dim_hidden, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln))

        # Pooling by multi-head attention
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads, cfg.num_features, ln=cfg.ln)

        self.dummy_param = nn.Parameter(torch.empty(0))

    def float2bit(self, f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        # SIGN BIT
        s = (torch.sign(f + 0.001) * -1 + 1) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        # EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        # MANTISSA
        f2 = f1 / 2 ** e_scientific
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=self.dummy_param.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=self.dummy_param.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, xx):
        if self.bit16:
            xx = self.float2bit(xx)
            xx = xx.view(xx.shape[0], xx.shape[1], -1)
            if self.norm:
                xx = (xx - 0.5) * 2
        if self.input_normalization:
            means = xx[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = xx[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            xx[:, :, -1] = (xx[:, :, -1] - means) / std

        if self.linear:
            if self.activation == 'relu':
                xx = torch.relu(self.linearl(xx))
            elif self.activation == 'sine':
                xx = torch.sin(self.linearl(xx))
            else:
                xx = (self.linearl(xx))

        xx = self.selfatt1(xx)
        for layer in self.selfatt:
            xx = layer(xx)
        xx = self.outatt(xx)
        return xx
