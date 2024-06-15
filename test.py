#Test with various hexdecimal inputs

import torch
from model import Transformer, PositionalEncoding  
from tokenizer import hex_encode, dec_encode, hex_pad_sequence, decode  
import pandas as pd  
from torch.utils.data import Dataset, DataLoader


def main():
    max_length = 15
    # Load the trained model
    model = Transformer(seq_length = 15, src_vocab_size = 17, tgt_vocab_size = 11, d_model = 512, nheads = 8, num_encoder_layers = 6, dropout = 0.1)
    model.load_state_dict(torch.load('transformer_model.pth')) #load weights
    model.eval()  # Set model to evaluation mode

    # Example input 
    example_input = "1A3F"  # Example hexadecimal input
    encoded_input = hex_encode(example_input)
    padded_input = hex_pad_sequence(encoded_input, max_length)

    # Convert input to torch tensor and add batch dimension
    input_tensor = torch.tensor(padded_input).unsqueeze(0)  # Shape: (1, max_length)

    # Perform inference
    with torch.no_grad():
        src_mask = None  # Define mask if needed
        output = model(input_tensor, src_mask)

    
    translated_sequence = decode(output.squeeze(0))  # Implement decode based on tokenizer

    print(f"Input: {example_input}")
    print(f"Translated Output: {translated_sequence}")

if __name__ == "__main__":
    main()