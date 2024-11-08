#File to generate data for the encoder
#Can modify number of data inputs

import random as random
import csv

def data_generation(num_samples):  
    hex_chars = '0123456789ABCDEF'
    data = []                     
    for _ in range(num_samples):  
        hex_num = ''.join(random.choices(hex_chars, k=random.randint(1, 8))) 
        dec_num = int(hex_num, 16) 
        dec_num = format(dec_num, ',') 
        dec_num = str(dec_num)  
        hex_num = '0x' + hex_num   
        data.append((hex_num, dec_num))  
    return data 

def create_csv(data, filename = "data.csv"):
    with open(filename, mode='w', newline='') as file: 
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)          
        writer.writerow(['hex', 'dec'])           
        writer.writerows(data)                     


if __name__ == "__main__":                   
    num_samples = 10000                 
    data = data_generation(num_samples)
    create_csv(data)
    print(f"Generated {num_samples} samples and saved to 'data.csv'")
