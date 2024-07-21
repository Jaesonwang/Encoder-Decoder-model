#The model architecture of the encoder
#Modelled based on the annotated attention is all you need paper

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #equation from the transformer paper
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, eps = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedforwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout)-> None:
        super().__init__()
        self.h = h
        self.d_model = d_model
        assert self.d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
        if mask is not None:
            mask = mask.to(attention_scores.dtype)  # Ensure mask has the same dtype as attention_scores
        
        # Ensure mask dimensions are compatible with attention_scores
            if mask.dim() < attention_scores.dim():
                mask = mask.unsqueeze(1)  # Add singleton dim to match attention_scores
        
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    
        attention_scores = attention_scores.softmax(dim=-1)
    
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = torch.matmul(attention_scores, value)
        return output, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) 
        
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, subLayer):
        return self.dropout(subLayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedforwardNetwork, dropout)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, src_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedforwardNetwork, dropout) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.src_attention_block = src_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.src_attention_block(x, memory, memory, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos_embd: PositionalEncoding, tgt_pos_embd: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos_embd = src_pos_embd
        self.tgt_pos_embd = tgt_pos_embd
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        embedded_src = self.src_embedding(src)
        encoder_inputs = self.src_pos_embd(embedded_src)
        return self.encoder(encoder_inputs, src_mask)
    
    def decode(self, encoder_output, tgt, src_mask, tgt_mask):
        embedded_tgt = self.tgt_embedding(tgt)
        decoder_inputs = self.tgt_pos_embd(embedded_tgt)
        return self.decoder(decoder_inputs, encoder_output, src_mask, tgt_mask)
    
    def proj_layer(self, x):
        return self.projection_layer(x)
    
def transformer_model(src_vocab_size, tgt_vocab_size, src_sequence_length, tgt_sequence_length, d_model, num_layers: int = 4 , num_heads: int = 4, dropout = 0.1, d_ff = 512) -> Transformer:
    
    src_embd = InputEmbedding(d_model, src_vocab_size)
    tgt_embd = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_sequence_length, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_sequence_length, dropout)

    complete_encoder = []
    complete_decoder = []
    
    for _ in range(num_layers):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        encoder_feed_forward = FeedforwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        complete_encoder.append(encoder_block)

        decoder_self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_feed_forward = FeedforwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
        complete_decoder.append(decoder_block)

    encoder = Encoder(nn.ModuleList(complete_encoder))
    decoder = Decoder(nn.ModuleList(complete_decoder))    

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embd, tgt_embd, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

