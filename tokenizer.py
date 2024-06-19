import pandas as pd
import torch
import torch.nn as nn


# Hexadecimal characters
hex_chars = '0123456789ABCDEF'
hex_vocab = sorted(set(hex_chars))
hex_vocab.append('<PAD>')
hex_char_to_index = {char: idx for idx, char in enumerate(hex_vocab)}
hex_index_to_char = {idx: char for idx, char in enumerate(hex_vocab)}


# Decimal characters
dec_chars = '0123456789'
dec_vocab = sorted(set(dec_chars))
dec_vocab.append('<PAD>')
dec_char_to_index = {char: idx for idx, char in enumerate(dec_vocab)}
dec_index_to_char = {idx: char for idx, char in enumerate(dec_vocab)}


def hex_encode(input_string):
    return [hex_char_to_index[char] for char in input_string]

def dec_encode(input_string):
    return [dec_char_to_index[char] for char in input_string]

def hex_pad_sequence(sequence, max_length, padding_value='<PAD>'):
    padding_length = max_length - len(sequence)
    return sequence + [hex_char_to_index[padding_value]] * padding_length

def dec_pad_sequence(sequence, max_length, padding_value='<PAD>'):
    padding_length = max_length - len(sequence)
    return sequence + [dec_char_to_index[padding_value]] * padding_length

# def decode(output_tensor): #Mock decoder function for testing
#     probabilities = torch.softmax(output_tensor, dim=-1)  
#     _, predicted_indices = torch.max(probabilities, dim=-1) 
#     predicted_indices = predicted_indices.squeeze().tolist()  
#     return ''.join([dec_index_to_char[idx] for idx in predicted_indices])

