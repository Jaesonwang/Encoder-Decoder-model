#File containing Training loop, validation and testing phases
#loss calculated is the cross entropy loss, will try to incorporate mean squared loss too
#contains model parameters -- can be modified
#data is split [training, validation, testing] == [80%, 10%, 10%]


import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import Transformer
from tokenizer import hex_encode, dec_encode, hex_pad_sequence,  dec_pad_sequence, hex_char_to_index, dec_index_to_char, dec_char_to_index

# Parameters
seq_length = 15  # Sequence length (assuming the length of padded sequences)
d_model = 128    # Model dimensionality
num_encoder_layers = 4   # Number of encoder layers
num_decoder_layers = 4   # Number of decoder layers
num_heads = 4    # Number of attention heads
dropout = 0.1    # Dropout rate
max_length = 15  # Maximum length for padding sequences
batch_size = 12  # Batch size
num_epochs = 20  # Number of epochs
learning_rate = 1e-3  # Learning rate
writer = SummaryWriter('logs')

# Dataset class
class HexDecDataset(Dataset):
    def __init__(self, csv_file, max_length):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.max_length = max_length 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hex_str = self.data.iloc[idx, 0]
        dec_str = self.data.iloc[idx, 1]
        
        hex_str = hex_str.replace('0x','')  #Removing prefix from hexadecimal number
        dec_str = dec_str.replace(',','')   #Removing commas from decimal number 
        
        hex_encoded = hex_encode(hex_str)  # Function to encode hexadecimal string
        dec_encoded = dec_encode(dec_str)  # Function to encode decimal string
        
        hex_padded = hex_pad_sequence(hex_encoded, self.max_length)  # Function to pad sequence
        dec_padded = dec_pad_sequence(dec_encoded, self.max_length)  # Function to pad sequence
        
        return torch.tensor(hex_padded), torch.tensor(dec_padded)


# Training function
def train():
    # Load dataset
    dataset = HexDecDataset('data.csv', max_length)
    trainDataSize = int(0.8 * len(dataset))
    valDataSize = int(0.1 * len(dataset))
    testDataSize = int(0.1 * len(dataset))

    trainDataset, valDataset, testDataset = random_split(dataset, [trainDataSize, valDataSize, testDataSize])

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    input_dim = len(hex_char_to_index)  #char_to_index is a dictionary mapping characters to indices
    output_dim = len(dec_index_to_char)
    src_padding_idx = hex_char_to_index['<PAD>']
    tgt_padding_idx = dec_char_to_index['<PAD>']
    
    model = Transformer(seq_length, input_dim, output_dim, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_padding_idx, label_smoothing=0.1)
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #create a src mask for input sequences
    def create_mask(src, tgt,  src_padding_idx, tgt_padding_idx):
        #src_mask = (src != src_padding_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        src_mask = (src != src_padding_idx).unsqueeze(-2)
        tgt_mask = (tgt != tgt_padding_idx).unsqueeze(-2)
        size = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type_as(tgt_mask)
        tgt_mask = tgt_mask & (nopeak_mask == 0)
        return src_mask, tgt_mask

    # Training loop
    
    for epoch in range(num_epochs):
        model.train()
        running_loss_CE = 0.0
        running_loss_MSE = 0.0
        for i, (inputs, targets) in enumerate(trainDataloader):
            
            optimizer.zero_grad()
            src_mask, tgt_mask = create_mask(inputs, targets, src_padding_idx, tgt_padding_idx)

            # Forward pass
            outputs = model(inputs, targets, src_mask, tgt_mask)
            
            # Calculate cross entropy loss
            outputs_CE = outputs.view(-1, output_dim)
            targets_CE = targets.contiguous().view(-1)
            loss_CE = criterion(outputs_CE, targets_CE)
            
            #Calculate Mean sqaured Loss
            outputs_for_mse = outputs.view(-1, output_dim)
            targets_for_mse = nn.functional.one_hot(targets, num_classes=output_dim).float().view(-1, output_dim)
            loss_MSE = criterion2(outputs_for_mse, targets_for_mse)

            loss_CE.backward() #Computes the gradient of the loss with respect to the model parameters.

            if i == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                    
            optimizer.step() #Updated the model parameters using the computed gradients 
            
            running_loss_CE += loss_CE.item() 
            running_loss_MSE += loss_MSE.item()

        epoch_loss_CE = running_loss_CE / len(trainDataloader)
        epoch_loss_MSE = running_loss_MSE / len(trainDataloader)
        print(f'Epoch {epoch+1}: Cross Entropy Loss = {epoch_loss_CE:.4f}, MSE = {epoch_loss_MSE:.4f}') #print loss after each epoch 

    # Save trained model
    print('Finished Training')
    torch.save(model.state_dict(), 'transformer_model.pth')

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valDataloader:
            src_mask, tgt_mask = create_mask(inputs, targets, src_padding_idx, tgt_padding_idx)
            outputs = model(inputs, targets, src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, output_dim), targets.contiguous().view(-1))
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(valDataloader):.4f}')

    # Testing step
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in testDataloader:
            src_mask, tgt_mask = create_mask(inputs, targets, src_padding_idx, tgt_padding_idx)
            outputs = model(inputs, targets, src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, output_dim), targets.contiguous().view(-1))
            test_loss += loss.item()

    print(f'Test Loss: {test_loss/len(testDataloader):.4f}')

if __name__ == "__main__": #For running the train.py file
    train()

