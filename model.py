import csv
import random
import math
import inspect
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

    def __init__(self, seq_len, d_model, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)# Register the positional encoding as a buffer

    def forward(self, x):
        #x should have size(batch_size, seq_len, d_model)
        #pe should have size(1, seq_len, d_model) 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
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

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
           attention_score = attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_score)

        return (attention_score @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores =MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) 
        
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

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    
    def __init__(self, seq_length, src_vocab_size, tgt_vocab_size, d_model, nheads, num_encoder_layers: int, dropout=0.1):
        super().__init__()
        self.src_embedding = InputEmbedding(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
        
        self.pos_encoder = PositionalEncoding(seq_length, d_model, dropout)

        self.encoder = Encoder([
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, nheads, dropout),
                FeedforwardNetwork(d_model, d_model * 4, dropout),
                dropout
            ) for _ in range(num_encoder_layers)
        ])

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.embedding.weight.data.uniform_(-initrange, initrange)
        self.projection_layer.proj.weight.data.uniform_(-initrange, initrange)
        self.projection_layer.proj.bias.data.zero_()

    def forward(self, src, src_mask):
        
        src = self.src_embedding(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, src_mask)
        output = self.projection_layer(output)
        return output
