import torch 
import torch.nn as nn
import secrets
from model import Transformer
from Train import seq_length, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout, max_length
from tokenizer import hex_encode, hex_pad_sequence, hex_char_to_index, dec_char_to_index, dec_index_to_char

model = Transformer(seq_length, 19, 13, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

def generate_random_hex(length):
    rand_int = secrets.randbits(length * 4)       # Multiply by 4 to convert bytes to hex digits
    hex_string = format(rand_int, f'0{length}x')  # Convert the integer to a hexadecimal string
    hex_string = hex_string.upper()
    return hex_string

random_hex = generate_random_hex(6)
hex_encoded = hex_encode(random_hex)
hex_padded = hex_pad_sequence(hex_encoded, max_length)
input_tensor = torch.tensor([hex_padded])

src_padding_idx = hex_char_to_index['<PAD>']
tgt_padding_idx = dec_char_to_index['<PAD>']

def create_mask(src, tgt, src_padding_idx, tgt_padding_idx):
    src_mask = (src != src_padding_idx).unsqueeze(-2)
    tgt_mask = (tgt != tgt_padding_idx).unsqueeze(-2)
    size = tgt.size(1)
    nopeak_mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type_as(tgt_mask)
    tgt_mask = tgt_mask & (nopeak_mask == 0)
    return src_mask, tgt_mask

dummy_targets = torch.zeros((1, seq_length), dtype=torch.long)

with torch.no_grad():
    src_mask, tgt_mask = create_mask(input_tensor, dummy_targets, src_padding_idx, tgt_padding_idx)
    print("src_mask:", src_mask)
    print("tgt_mask:", tgt_mask)
    outputs = model(input_tensor, dummy_targets, src_mask, tgt_mask)
    print("outputs:", outputs)

_, predicted_indices = torch.max(outputs, dim=-1)
print("predicted_indices:", predicted_indices)

predicted_decoded = [dec_index_to_char[idx.item()] for idx in predicted_indices.squeeze()]
translated_output = ''.join(predicted_decoded).strip('<PAD>')  # Remove padding tokens

print(f"Input Hexadecimal: {random_hex}")
print(f"Predicted Decimal: {translated_output}")

