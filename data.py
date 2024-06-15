import numpy
import random as random
import csv

def generate_data(num_samples):
    hex_chars = '0123456789ABCDEF'
    data = []
    
    for _ in range(num_samples):
        hex_num = ''.join(random.choices(hex_chars, k=random.randint(1, 8)))
        dec_num = int(hex_num, 16)
        #dec_num = format(dec_num, ',')
        dec_num = str(dec_num)
        #dec_num = f"{dec_num:,}"
        #hex_num = '0x' + hex_num
        data.append((hex_num, dec_num))

    return data

def create_csv(data, filename = "data.csv"):
    with open(filename, mode='w', newline='') as file: #add 0x in front of hexadecimal 
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)                #add comma in decimal number
        writer.writerow(['hex', 'dec'])
        writer.writerows(data)


if __name__ == "__main__":
    num_samples = 100
    data = generate_data(num_samples)
    #  print(data)
    create_csv(data)
    print(f"Generated {num_samples} samples and saved to 'data.csv'")