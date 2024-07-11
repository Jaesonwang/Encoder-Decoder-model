import torch
import torch.nn as nn
from model import Transformer
from Train import seq_length, d_model, num_heads, num_layers, dropout
from tokenizer import hex_encode, hex_pad_sequence, dec_index_to_char, dec_char_to_index

model = Transformer(seq_length, 19, 13, d_model, num_heads, num_layers, dropout)
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

def translate_hex_to_dec(hex_input):
    with torch.no_grad():
        
        hex_encoded = hex_encode(hex_input)
        hex_padded = hex_pad_sequence(hex_encoded, seq_length)
        print(hex_padded)
        hex_tensor = torch.tensor(hex_padded).unsqueeze(0)  
        
        src_mask = None  

        outputs = model(hex_tensor, src_mask)
        _, predicted_indices = torch.max(outputs, dim=-1)

        predicted_decimal = []
        for idx in predicted_indices.squeeze():
            idx_item = idx.item()
            if idx_item == dec_char_to_index['<PAD>']:  #stop translating when padding token is reached
                break
            predicted_decimal.append(dec_index_to_char[idx_item])

        predicted_decimal = ''.join(predicted_decimal)
        
        predicted_decimal = format(int(predicted_decimal), ',')
        hex_input = "0x" + hex_input
        

        print(f'Hexadecimal Input: {hex_input}')
        print(f'Predicted Decimal Output: {predicted_decimal}')
        hex_input = hex_input.replace('0x','')
        # print(f'Expected Decimal Output: {format(int(hex_input, 16), ',')}')
        print("------------------------------")
    return predicted_decimal

translate_hex_to_dec("1AB1212432121")
translate_hex_to_dec("1")
translate_hex_to_dec("2")
translate_hex_to_dec("3")
translate_hex_to_dec("4")
translate_hex_to_dec("5")
translate_hex_to_dec("6")
translate_hex_to_dec("7")