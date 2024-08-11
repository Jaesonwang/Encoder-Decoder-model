import torch

from tokenizer import CharTokenizer
from model import transformer_model
from train import greedy_decode, ModelConfig, causal_mask, encoder_decoder, get_weights_file_path, find_latest_checkpoint

def convert_hex_to_dec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizers
    hex_tokenizer = CharTokenizer(vocab=list("0123456789ABCDEF") + ["<pad>", "<eos>", "<bos>", "<unk>"])
    dec_tokenizer = CharTokenizer(vocab=list("0123456789") + ["<pad>", "<eos>", "<bos>", "<unk>"])

    # Token IDs for <bos> and <eos>
    sos_token_id = dec_tokenizer._convert_token_to_id('<bos>')
    eos_token_id = dec_tokenizer._convert_token_to_id('<eos>')

    # Initialize model
    model = encoder_decoder(len(hex_tokenizer.vocab), len(dec_tokenizer.vocab)).to(device)
    
    #ModelConfig.weight_folder = "_Weights(300000data)"

    # Load the latest model weights
    latest_epoch = find_latest_checkpoint(ModelConfig.weight_folder, ModelConfig.weight_file_name_base)
    if latest_epoch is not None:
        model_filename = get_weights_file_path(f"{latest_epoch:02d}")
        if torch.cuda.is_available():
            state = torch.load(model_filename)
        else:
            state = torch.load(model_filename, map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded model weights from {model_filename}")

    model.eval()

    while True:
        hex_input = input("Provide a hexadecimal value up to 10 characters (or type 'exit' to quit): ")
        
        if hex_input.lower() == 'exit':
            break

        # Encode and pad the hexadecimal input
        hex_encoded = [hex_tokenizer._convert_token_to_id(token) for token in hex_input]
        hex_padded = hex_tokenizer.hex_pad_sequence(hex_encoded, ModelConfig.max_length)
        encoder_mask = (torch.tensor(hex_padded).unsqueeze(0) != hex_tokenizer.pad_token_id).unsqueeze(0).int()

        # Convert using the greedy decoder
        with torch.no_grad():
            model_out = greedy_decode(model, torch.tensor(hex_padded).unsqueeze(0).to(device), encoder_mask.to(device), device, sos_token_id, eos_token_id)
        model_out_text = dec_tokenizer.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)

        print(f"Decimal value: {model_out_text}")
        print(f"Expected decimal value should be: {int(hex_input, 16)}")

if __name__ == '__main__':
    convert_hex_to_dec()
    