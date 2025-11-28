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
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
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
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
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
        """Generate prediction for a single sequence (greedy decoding)"""
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

    def beam_search(self, src, sos_token, eos_token, beam_width=5, max_length=20, length_penalty=0.6):
        """
        🆕 Beam Search Decoding - More accurate than greedy!

        Args:
            src: source pose sequence (seq_len, input_dim)
            sos_token: start of sequence token ID
            eos_token: end of sequence token ID
            beam_width: number of beams to keep (default=5)
            max_length: maximum sequence length
            length_penalty: penalty for longer sequences (default=0.6)

        Returns:
            best_sequence: most likely token sequence
        """
        self.eval()

        with torch.no_grad():
            # Encode source once
            memory = self.encoder(src.unsqueeze(1))  # (seq_len, 1, d_model)

            # Initialize beams: [(sequence, score)]
            beams = [(torch.LongTensor([[sos_token]]).to(src.device), 0.0)]

            for step in range(max_length - 1):
                all_candidates = []

                for seq, score in beams:
                    # Skip if this beam already ended
                    if seq[0, -1].item() == eos_token:
                        all_candidates.append((seq, score))
                        continue

                    # Generate mask
                    seq_len = seq.size(1)
                    tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

                    # Decode
                    seq_transposed = seq.transpose(0, 1)  # (seq_len, 1)
                    out = self.decoder(seq_transposed, memory, tgt_mask=tgt_mask)

                    # Get log probabilities for next token
                    log_probs = F.log_softmax(out[-1, 0, :], dim=-1)

                    # Get top k candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    # Create new candidates
                    for log_prob, idx in zip(top_log_probs, top_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + log_prob.item()
                        all_candidates.append((new_seq, new_score))

                # Select top beam_width candidates
                # Apply length penalty: score / (length ** length_penalty)
                ordered = sorted(all_candidates,
                               key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                               reverse=True)
                beams = ordered[:beam_width]

                # Check if all beams ended
                if all(seq[0, -1].item() == eos_token for seq, _ in beams):
                    break

            # Return best sequence
            best_seq, best_score = beams[0]
            return best_seq.squeeze(0)


def test_beam_search():
    """Test beam search implementation"""
    print("Testing Beam Search...")

    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=41,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )

    # Create dummy input
    src = torch.randn(60, 375)  # 60 frames, 375 features
    sos_token = 37
    eos_token = 38

    # Test greedy decoding
    greedy_output = model.predict(src, sos_token, eos_token)
    print(f"Greedy output: {greedy_output.tolist()}")

    # Test beam search
    beam_output = model.beam_search(src, sos_token, eos_token, beam_width=5)
    print(f"Beam search output: {beam_output.tolist()}")

    print("✅ Beam search working!")


if __name__ == "__main__":
    test_beam_search()