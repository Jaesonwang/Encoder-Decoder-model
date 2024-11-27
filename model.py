# #The model architecture of the encoder
# #Modelled based on the annotated attention is all you need paper

import torch
import torch.nn as nn
import math

#-------------------------------------------------------------------------------------------------------------------------

#Encoder-decoder components

class InputEmbeddingLayer(nn.Module): 

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  
    
class PositionalEncodingLayer(nn.Module): 

    def __init__(self, d_model, sequence_length, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias=False) 
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False) 
        self.w_o = nn.Linear(d_model, d_model, bias=False) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k) 
        value = self.w_v(v) 

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class LayerNorm(nn.Module):  

    def __init__(self, features, eps = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) 
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module): 

    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    
        def __init__(self, features, dropout) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNorm(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    

#-------------------------------------------------------------------------------------------------------------------------

#Building the Encoder and Decoder


class Encoder(nn.Module):
    
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            self._build_layer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def _build_layer(self, d_model, num_heads, d_ff, dropout):
        layer = nn.ModuleDict({
            'self_attention_block': MultiHeadAttentionBlock(d_model, num_heads, dropout),
            'feed_forward_block': FeedForwardBlock(d_model, d_ff, dropout),
            'residual_connections': nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
        })
        return layer

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer['residual_connections'][0](x, lambda x: layer['self_attention_block'](x, x, x, src_mask))
            x = layer['residual_connections'][1](x, layer['feed_forward_block'])
        return self.norm(x)
    
class Decoder(nn.Module):

    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            self._build_layer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def _build_layer(self, d_model, num_heads, d_ff, dropout):
        layer = nn.ModuleDict({
            'self_attention_block': MultiHeadAttentionBlock(d_model, num_heads, dropout),
            'cross_attention_block': MultiHeadAttentionBlock(d_model, num_heads, dropout),
            'feed_forward_block': FeedForwardBlock(d_model, d_ff, dropout),
            'residual_connections': nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        })
        return layer

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer['residual_connections'][0](x, lambda x: layer['self_attention_block'](x, x, x, tgt_mask))
            x = layer['residual_connections'][2](x, lambda x: layer['cross_attention_block'](x, encoder_output, encoder_output, src_mask))
            x = layer['residual_connections'][1](x, layer['feed_forward_block'])
        return self.norm(x)
    
#-------------------------------------------------------------------------------------------------------------------------

#Building the full transformer model
    
class TransformerFunctions(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddingLayer, tgt_embed: InputEmbeddingLayer, src_pos: PositionalEncodingLayer, tgt_pos: PositionalEncodingLayer, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def transformer_model(src_vocab_size, tgt_vocab_size, src_sequence_length, tgt_sequence_length, d_model = 128, num_layers = 4, num_heads = 4, dropout = 0.1, d_ff = 512) -> TransformerFunctions:
    
    encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
    decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)    
    
    src_embd = InputEmbeddingLayer(d_model, src_vocab_size)
    tgt_embd = InputEmbeddingLayer(d_model, tgt_vocab_size)
    src_pos = PositionalEncodingLayer(d_model, src_sequence_length, dropout)
    tgt_pos = PositionalEncodingLayer(d_model, tgt_sequence_length, dropout)
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = TransformerFunctions(encoder, decoder, src_embd, tgt_embd, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer



