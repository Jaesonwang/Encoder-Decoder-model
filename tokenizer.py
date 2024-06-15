import pandas as pd
import torch
import torch.nn as nn


hex_vocab = '0123456789ABCDEF'
dec_vocab = '0123456789'

vocab = sorted(set(dec_vocab + hex_vocab))

vocab.append('<PAD>')

char_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_char = {idx: char for idx, char in enumerate(vocab)}


# def encode(input_string):
#     return [char_to_index.get(char, char_to_index['<PAD>']) for char in input_string]

def encode(input_string):
    if isinstance(input_string, str):
        return [char_to_index.get(char, char_to_index['<PAD>']) for char in input_string]
    elif isinstance(input_string, int):
        input_string = str(input_string)  # Convert int to string
        return [char_to_index.get(char, char_to_index['<PAD>']) for char in input_string]
    else:
        raise TypeError("Input should be a string or integer.")

def decode(input_indicies):
    return ''.join([index_to_char[idx] for idx in input_indicies])

def pad_sequence(sequence, max_length, padding_value='<PAD>'):
    padding_length = max_length - len(sequence)
    return sequence + [char_to_index[padding_value]] * padding_length

