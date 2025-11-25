import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PoseEncoder(nn.Module):
    def __init__(self, input_dim=375, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super(PoseEncoder, self).__init__()

        self.d_model = d_model

        # Add input normalization (IMPORTANT for pose data)
        self.input_norm = nn.LayerNorm(input_dim)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # More proportional (1024 instead of 2048)
            dropout=dropout,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=False,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )

    def forward(self, src):
        # src shape: (seq_len, batch_size, input_dim)

        # Normalize input first
        src = src.transpose(0, 1)  # (batch, seq, features)
        src = self.input_norm(src)
        src = src.transpose(0, 1)  # back to (seq, batch, features)

        # Project input to model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)
        src = self.dropout(src)

        # Transformer encoder
        memory = self.transformer_encoder(src)

        return memory


class GlossDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.2, max_seq_length=20):
        super(GlossDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # More proportional
            dropout=dropout,
            activation='gelu',  # GELU instead of ReLU
            batch_first=False,
            norm_first=True  # Pre-norm
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

        # WEIGHT TYING: Share weights between embedding and output
        self.output_projection.weight = self.embedding.weight

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt shape: (seq_len, batch_size)
        # memory shape: (src_seq_len, batch_size, d_model)

        # Embed target tokens
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        output = self.output_projection(output)

        return output


class BanglaSignTransformer(nn.Module):
    def __init__(self, input_dim=375, vocab_size=41, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2, dropout=0.2, max_seq_length=7):
        super(BanglaSignTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Encoder for pose sequences
        self.encoder = PoseEncoder(input_dim, d_model, nhead, num_encoder_layers, dropout)

        # Decoder for gloss sequences
        self.decoder = GlossDecoder(vocab_size, d_model, nhead, num_decoder_layers, dropout, max_seq_length)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for transformer decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src: pose sequence (seq_len, batch_size, input_dim)
        # tgt: target gloss sequence (tgt_seq_len, batch_size)

        # Encode pose sequence
        memory = self.encoder(src)

        # Generate mask for decoder
        tgt_seq_len = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Decode to gloss sequence
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return output

    def predict(self, src, sos_token, eos_token, max_length=20):
        """Generate prediction for a single sequence (greedy decoding) - IMPROVED"""
        self.eval()

        with torch.no_grad():
            # Encode source
            memory = self.encoder(src.unsqueeze(1))  # Add batch dimension

            # Initialize target with SOS token
            ys = torch.ones(1, 1).fill_(sos_token).long().to(src.device)

            for i in range(max_length - 1):
                # Generate mask for current sequence
                tgt_mask = self.generate_square_subsequent_mask(ys.size(0)).to(src.device)

                # Decode
                out = self.decoder(ys, memory, tgt_mask=tgt_mask)

                # Get next token (greedy)
                next_token = out[-1, :, :].argmax(dim=-1).unsqueeze(0)

                # Stop if EOS token is generated
                if next_token.item() == eos_token:
                    ys = torch.cat([ys, next_token], dim=0)
                    break

                ys = torch.cat([ys, next_token], dim=0)

            return ys.squeeze(1)