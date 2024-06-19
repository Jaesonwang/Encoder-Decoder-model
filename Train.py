import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import Transformer, PositionalEncoding
from tokenizer import hex_encode, dec_encode, hex_pad_sequence,  dec_pad_sequence, hex_char_to_index, dec_index_to_char

# Parameters
seq_length = 15  # Sequence length (assuming the length of padded sequences)
d_model = 512    # Model dimensionality
num_layers = 6   # Number of encoder layers
num_heads = 8    # Number of attention heads
dropout = 0.1    # Dropout rate
max_length = 15  # Maximum length for padding sequences
batch_size = 32  # Batch size
num_epochs = 20  # Number of epochs
learning_rate = 1e-3  # Learning rate

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
    
    #print(seq_length, input_dim, output_dim, d_model, num_heads, num_layers, dropout)
    
    model = Transformer(seq_length, input_dim, output_dim, d_model, num_heads, num_layers, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainDataloader):
            
            optimizer.zero_grad()
            src_mask = None  # May need to define a mask if needed
            
            # Forward pass
            
            outputs = model(inputs, src_mask)
            
            
            # Calculate loss
            loss = criterion(outputs.view(-1, output_dim), targets.contiguous().view(-1))
            

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % trainDataSize == 0: 
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainDataloader)}], Loss: {running_loss/trainDataSize:.4f}') #change 100 to num of inputs
                running_loss = 0.0


        print(f'Loss = {running_loss}')
       

    # Save trained model
    print('Finished Training')
    torch.save(model.state_dict(), 'transformer_model.pth')

    #Add Validation
    
    #Add testing

    #In a batch of 1000 data points, split (800,100,100)

if __name__ == "__main__": #For running the train.py file
    train()

