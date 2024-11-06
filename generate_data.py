#File to generate data for the encoder
#Can modify number of data inputs

import random as random
import csv

def data_generation(num_samples):    #generate data for encoder
    hex_chars = '0123456789ABCDEF' #vocabulary for hexadecimals
    data = []                      #create array for data
    
    for _ in range(num_samples):   #create num_samples amount of data
        hex_num = ''.join(random.choices(hex_chars, k=random.randint(1, 8)))  # generate random hexadecimal values that range from 1-8 digits
        dec_num = int(hex_num, 16) # convert to decimal value
        dec_num = format(dec_num, ',')  # --- include to add commas for better clarity for decimal values
        dec_num = str(dec_num)  # convert to decimal value to string
        hex_num = '0x' + hex_num   #add 0x prefix in front of hexadecimals 
        data.append((hex_num, dec_num))  #add to array 
        
    return data #return array

def create_csv(data, filename = "data.csv"):    #create the csv file to store data
    with open(filename, mode='w', newline='') as file: #add 0x in front of hexadecimal 
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)                #add comma in decimal number
        writer.writerow(['hex', 'dec'])            #header for the columns in the csv
        writer.writerows(data)                      #add data to the csv


if __name__ == "__main__":                      #run the data.py file
    num_samples = 10000                 #controls the amount of data generated
    data = data_generation(num_samples)
    create_csv(data)
    print(f"Generated {num_samples} samples and saved to 'data.csv'")
