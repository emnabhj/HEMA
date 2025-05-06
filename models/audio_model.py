import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=1024, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm_mha = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_mha = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(dim)
        self.conv_module = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x_norm = self.norm_mha(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout_mha(attn_output)
        x_norm = self.norm_conv(x)
        x_conv = self.conv_module(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AudioFrontEnd(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=512,
                 num_layers=6, num_heads=4, dropout=0.1):
        super(AudioFrontEnd, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, hidden_dim * 4, 31, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        x = self.input_projection(x)
        x = self.pos_enc(x)
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            batch_size = x.size(0)
            mask = torch.arange(max_len).expand(batch_size, max_len).to(x.device) < lengths.unsqueeze(1)
        for block in self.conformer_blocks:
            x = block(x)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        return self.output_projection(x)


class AudioModel(nn.Module):
    """
    Wrapper class for the audio front-end to match the interface expected by the HEMA model.
    """
    def __init__(self, hparams):
        super(AudioModel, self).__init__()
        self.audio_net = AudioFrontEnd(
            input_dim=hparams.audio_feat_dim,
            hidden_dim=hparams.audio_hidden_dim,
            output_dim=hparams.fusion_dim,
            num_layers=hparams.audio_num_layers,
            num_heads=hparams.audio_num_heads,
            dropout=hparams.dropout
        )

    def forward(self, audio_feats):
        return self.audio_net(audio_feats)
