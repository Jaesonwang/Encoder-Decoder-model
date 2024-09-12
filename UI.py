import tkinter as tk
from tkinter import messagebox
from converter import HexToDecConverter

class HexToDecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hexadecimal to Decimal Converter")
        self.converter = HexToDecConverter()

        # UI Elements
        self.label = tk.Label(root, text="Enter a hexadecimal value:")
        self.label.pack(pady=10)

        self.hex_entry = tk.Entry(root, width=40)
        self.hex_entry.pack(pady=5)

        self.convert_button = tk.Button(root, text="Convert", command=self.convert_hex_to_dec)
        self.convert_button.pack(pady=20)

        self.result_label = tk.Label(root, text="Decimal value will be displayed here.")
        self.result_label.pack(pady=10)

        self.expected_label = tk.Label(root, text="Expected Decimal Value:")
        self.expected_label.pack(pady=5)    


    def convert_hex_to_dec(self):
        hex_input = self.hex_entry.get().strip().upper()
        
        if len(hex_input) == 0 or len(hex_input) > 12:
            messagebox.showerror("Invalid Input", "Please enter a valid hexadecimal value (1 to 12 characters).")
            return

        try:
            translated_text = self.converter.convert_hex_to_dec(hex_input)
            self.result_label.config(text=f"Decimal Value: {translated_text}")
            self.expected_label.config(text=f"Expected Value: {int(hex_input, 16)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = HexToDecApp(root)
    root.mainloop()