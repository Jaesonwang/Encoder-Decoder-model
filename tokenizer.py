#Tokenizer Class


class CharTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = ["<pad>", "<eos>", "<bos>", "<unk>"] + list("0123456789ABCDEF")
        self.vocab = vocab
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}
        self.pad_token_id = self.char2idx["<pad>"]

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.char2idx.get(token, self.char2idx["<unk>"])

    def _convert_id_to_token(self, index):
        return self.idx2char.get(index, "<unk>")

    def hex_pad_sequence(self, sequence, max_length, padding_value='<pad>', SOS_value='<bos>', EOS_value='<eos>'):
        padding_value_id = self.char2idx[padding_value]
        SOS_value_id = self.char2idx[SOS_value]
        EOS_value_id = self.char2idx[EOS_value]

        padding_length = max_length - len(sequence) - 2
        return [SOS_value_id] + sequence + [EOS_value_id] + [padding_value_id] * padding_length

    def dec_pad_sequence(self, sequence, max_length, padding_value='<pad>', SOS_value='<bos>'):
        padding_value_id = self.char2idx[padding_value]
        SOS_value_id = self.char2idx[SOS_value]

        padding_length = max_length - len(sequence) - 1
        return [SOS_value_id] + sequence + [padding_value_id] * padding_length

    def label_pad_sequence(self, sequence, max_length, padding_value='<pad>', EOS_value='<eos>'):
        padding_value_id = self.char2idx[padding_value]
        EOS_value_id = self.char2idx[EOS_value]

        padding_length = max_length - len(sequence) - 1
        return sequence + [EOS_value_id] + [padding_value_id] * padding_length

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self._convert_id_to_token(idx) for idx in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in ["<pad>", "<eos>", "<bos>", "<unk>"]]
        return "".join(tokens)