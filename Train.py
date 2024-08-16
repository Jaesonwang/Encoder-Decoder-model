import os
import torch
import torch.nn as nn
import pandas as pd
import tqdm
import warnings
import shutil
import re

from tokenizer import CharTokenizer
from model import transformer_model
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore

#Dataclass containing parameters

@dataclass
class ModelConfig:
    batch_size: int = 12
    num_epochs: int = 20
    learning_rate: float = 1e-3
    sequence_length: int = 15
    max_length: int = 15
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    weight_folder: str = "Weights"
    weight_file_name_base: str = "Transformer_weight_epoch"
    preload: str = None
    experiment_name: str = "runs"

#-------------------------------------------------------------------------------------------------------------------------------

#Helper functions
    

def get_weights_file_path(epoch):
    weight_folder_path = Path('.') / ModelConfig.weight_folder
    # if not weight_folder_path.exists():
    #     weight_folder_path.mkdir(parents=True, exist_ok=True)
    return str(weight_folder_path / f"{ModelConfig.weight_file_name_base}{epoch}.pt")

def get_console_width():
    try:
        size = shutil.get_terminal_size()
        return size.columns
    except Exception as e:
        print(f"Could not get terminal size: {e}")
        return 80 
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def find_latest_checkpoint(directory, basename):
    weight_folder_path = Path('.') / ModelConfig.weight_folder
    if not weight_folder_path.exists():
        weight_folder_path.mkdir(parents=True, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(directory) if re.match(rf'{basename}\d+\.pt', f)]
    if not checkpoint_files:
        return None
    epochs = [int(re.search(r'(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    return latest_epoch

def preload_model(model, optimizer):
    initial_epoch = 0
    num_step = 0

    latest_epoch = find_latest_checkpoint(ModelConfig.weight_folder, ModelConfig.weight_file_name_base)
    
    if latest_epoch is not None:
        if latest_epoch == ModelConfig.num_epochs:
            print(f"Maximum epochs reached in {ModelConfig.weight_folder} folder.")
            print("You may delete the weights folder or rename it to something else to retrain the model from scratch.")
            return -1, -1
        else:    
            print(f'Resuming training at Epoch {(latest_epoch + 1):02d}')
            ModelConfig.preload = latest_epoch 

    if ModelConfig.preload:
        model_filename = get_weights_file_path(f"{ModelConfig.preload:02d}")
        if os.path.exists(model_filename):
            print(f'Preloading model weights from {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            num_step = state['num_step']
            del state
        else:
            print(f"Checkpoint file not found: {model_filename}. Starting from scratch.")
    else:
        print("No preloaded weights exists. Starting from scratch.")
        
    return initial_epoch, num_step
#-------------------------------------------------------------------------------------------------------------------------------

#Collect and prepare data from dataset
#Create custom tokenizer

class HexDecDataset(Dataset):
    def __init__(self, csv_file, src_tokenizer, tgt_tokenizer, max_length):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hex_str = self.data.iloc[idx, 0]
        dec_str = self.data.iloc[idx, 1]

        hex_str = hex_str.replace('0x', '')
        dec_str = dec_str.replace(',', '')

        hex_encoded = [self.src_tokenizer._convert_token_to_id(token) for token in hex_str]
        dec_encoded = [self.tgt_tokenizer._convert_token_to_id(token) for token in dec_str]

        hex_padded = self.src_tokenizer.hex_pad_sequence(hex_encoded, self.max_length)
        dec_padded = self.tgt_tokenizer.dec_pad_sequence(dec_encoded, self.max_length)
        label = self.tgt_tokenizer.label_pad_sequence(dec_encoded, self.max_length)

        encoder_mask = (torch.tensor(hex_padded) != self.src_tokenizer.pad_token_id).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (torch.tensor(dec_padded) != self.tgt_tokenizer.pad_token_id).unsqueeze(0).unsqueeze(0).int() & \
                       causal_mask(torch.tensor(dec_padded).size(0))

        return {
            "encoder_input": torch.tensor(hex_padded),
            "decoder_input": torch.tensor(dec_padded),
            "label": torch.tensor(label),
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "src_value": hex_str,
            "tgt_value": dec_str,
        }

def collate_fn(batch): 
    encoder_inputs = torch.stack([item['encoder_input'] for item in batch])
    decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    encoder_masks = torch.stack([item['encoder_mask'] for item in batch])
    decoder_masks = torch.stack([item['decoder_mask'] for item in batch])
    src_values = [item['src_value'] for item in batch]  
    tgt_values = [item['tgt_value'] for item in batch]  

    return {
        "encoder_input": encoder_inputs,
        "decoder_input": decoder_inputs,
        "label": labels,
        "encoder_mask": encoder_masks,
        "decoder_mask": decoder_masks,
        "src_value": src_values,
        "tgt_value": tgt_values,
    }

def get_data_and_tokenizer():
    src_tokenizer = CharTokenizer(vocab=list("0123456789ABCDEF") + ["<pad>", "<eos>", "<bos>", "<unk>"])
    tgt_tokenizer = CharTokenizer(vocab=list("0123456789") + ["<pad>", "<eos>", "<bos>", "<unk>"])
    
    dataset = HexDecDataset('data.csv', src_tokenizer, tgt_tokenizer, ModelConfig.max_length)

    train_data_size = int(0.9 * len(dataset))
    val_data_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_data_size, val_data_size])

    train_dataloader = DataLoader(train_ds, batch_size=ModelConfig.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

#-------------------------------------------------------------------------------------------------------------------------------

#Build the transformer 

def encoder_decoder(src_vocab_length, tgt_vocab_length):
    model = transformer_model(src_vocab_length, tgt_vocab_length, ModelConfig.sequence_length, ModelConfig.sequence_length, ModelConfig.d_model)
    return model

#-------------------------------------------------------------------------------------------------------------------------------

#Functions to train and validate model

def run_training_loop(model, device, loss_fn, tgt_tokenizer, optimizer, num_step, batch_iterator, writer):
    
    for batch in batch_iterator:
        
        #Encoder Processes
        encoder_input = batch['encoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        encoder_output = model.encode(encoder_input, encoder_mask)
        
        #Decoder Process
        decoder_input = batch['decoder_input'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

        #Project decoder Outputs
        proj_output = model.project(decoder_output)

        #Compare projected outputs to expected outputs
        label = batch['label'].to(device)
        loss = loss_fn(proj_output.view(-1, len(tgt_tokenizer.vocab)), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Log to TensorBoard
        writer.add_scalar('train/loss', loss.item(), num_step)
        
        #Backward propagation and update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        num_step += 1

def greedy_decode(model, source, source_mask, device, tgt_sos_token_id, tgt_eos_token_id):

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(tgt_sos_token_id).type_as(source).to(device)
    step = 0
    
    while step < ModelConfig.max_length:

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word.item() == tgt_eos_token_id:
            break

        step += 1

    return decoder_input.squeeze(0)

def validation_step(model, val_dataset, tgt_tokenizer, device, num_step, writer):
    model.eval()

    sos_token_id = tgt_tokenizer._convert_token_to_id('<bos>')
    eos_token_id = tgt_tokenizer._convert_token_to_id('<eos>')

    source = []
    expected = []
    predicted = []
    example_num = 0
    num_examples = 2
    matching_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch in val_dataset:
            example_num += 1

            source_text = batch["src_value"][0]
            target_text = batch["tgt_value"][0]
            source.append(source_text)
            expected.append(target_text)
            
            #Get decoded output from greedy decode
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device) 
            model_out = greedy_decode(model, encoder_input, encoder_mask, device, sos_token_id, eos_token_id)
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)
            predicted.append(model_out_text)
            
            print('-'*get_console_width())
            print(f"{f'EXAMPLE {example_num}:':>11}")
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

            if example_num == num_examples:
                print('-'*get_console_width())
                break

    for pred, exp in zip(predicted, expected):
        min_len = min(len(pred), len(exp))
        total_chars += min_len
        matching_chars += sum(1 for p, e in zip(pred, exp) if p == e)

    accuracy = matching_chars / total_chars
    writer.add_scalar('validation/accuracy', accuracy, num_step)
    print(f"Validation accuracy: {accuracy:.3f}")
    print('-'*get_console_width())

    metric = BLEUScore()
    bleu = metric(predicted, expected)
    writer.add_scalar('validation/bleu', bleu, num_step)

#-------------------------------------------------------------------------------------------------------------------------------

#Actual Training Loop

def train():

    #Looks for CUDA, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Get data, tokenizer and model
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_data_and_tokenizer()
    model = encoder_decoder(len(src_tokenizer.vocab), len(tgt_tokenizer.vocab)).to(device)

    #Get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.learning_rate, eps=1e-9)

    #Preload model with pretrained weights if any, else start from scratch
    initial_epoch, num_step = preload_model(model, optimizer)

    #Break out of program if max epochs are reached in pretrained weights file
    if initial_epoch == -1:
        return

    #cross entropy loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_token_id, label_smoothing=0.1).to(device)

    #tensorboard for logging
    writer = SummaryWriter(log_dir=ModelConfig.experiment_name)
    
    #Epoch loop
    for epoch in range(initial_epoch, ModelConfig.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {(epoch + 1):02d}")
        
        run_training_loop(model, device, loss_fn, tgt_tokenizer, optimizer, num_step, batch_iterator, writer)
        validation_step(model, val_dataloader, tgt_tokenizer, device, num_step, writer)

        model_filename = get_weights_file_path(f"{(epoch + 1):02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_step': num_step
        }, model_filename)

    writer.close()

#-------------------------------------------------------------------------------------------------------------------------------

#Runs this when calling file

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    ModelConfig.preload = None
    
    train()