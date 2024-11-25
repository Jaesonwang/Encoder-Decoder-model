import torch
from tokenizer import CharTokenizer
from Train import ModelConfig, causal_mask, encoder_decoder, get_weights_file_path, find_latest_checkpoint

class HexToDecConverter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hex_tokenizer = CharTokenizer(vocab=list("0123456789ABCDEF") + ["<pad>", "<eos>", "<bos>", "<unk>"])
        self.dec_tokenizer = CharTokenizer(vocab=list("0123456789") + ["<pad>", "<eos>", "<bos>", "<unk>"])

        self.bos_token_id = self.dec_tokenizer._convert_token_to_id('<bos>')
        self.eos_token_id = self.dec_tokenizer._convert_token_to_id('<eos>')

        self.model = encoder_decoder(len(self.hex_tokenizer.vocab), len(self.dec_tokenizer.vocab)).to(self.device)
        self.model.eval()

        latest_epoch = find_latest_checkpoint(ModelConfig.weight_folder, ModelConfig.weight_file_name_base)
        if latest_epoch is not None:
            model_filename = get_weights_file_path(f"{latest_epoch:02d}")
            if torch.cuda.is_available():
                state = torch.load(model_filename)
            else:
                state = torch.load(model_filename, map_location=torch.device('cpu'))
            self.model.load_state_dict(state['model_state_dict'])

    def beam_search(self, model, source, source_mask, device, bos_token_id, eos_token_id, beam_width=3):
        encoder_output = model.encode(source, source_mask)
        beam = [(torch.tensor([[bos_token_id]], device=device), 0.0)]
        
        for step in range(ModelConfig.max_length):
            candidates = []
            for seq, score in beam:
                decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, seq, decoder_mask)
                prob = model.project(out[:, -1])
                topk_prob, topk_token = torch.topk(prob, beam_width, dim=1)
                for i in range(beam_width):
                    next_seq = torch.cat([seq, torch.tensor([[topk_token[0, i]]], device=device)], dim=1)
                    next_score = score + torch.log(topk_prob[0, i])
                    candidates.append((next_seq, next_score))
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[0, -1].item() == eos_token_id for seq, _ in beam):
                break
        return beam[0][0].squeeze(0)

    def convert_hex_to_dec(self, hex_input):
        hex_encoded = [self.hex_tokenizer._convert_token_to_id(token) for token in hex_input]
        hex_padded = torch.tensor(self.hex_tokenizer.hex_pad_sequence(hex_encoded, ModelConfig.max_length)).unsqueeze(0).to(self.device)
        encoder_mask = (hex_padded != self.hex_tokenizer.pad_token_id).unsqueeze(0).int().to(self.device)

        with torch.no_grad():
            translated_seq = self.beam_search(self.model, hex_padded, encoder_mask, self.device, self.bos_token_id, self.eos_token_id, beam_width=10)
        translated_text = self.dec_tokenizer.decode(translated_seq.cpu().numpy(), skip_special_tokens=True)

        return translated_text
    
    