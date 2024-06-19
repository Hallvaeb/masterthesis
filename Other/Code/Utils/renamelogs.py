import os
import re

def rename_files():
    # Define the regular expression pattern to match the files
    # pattern = re.compile(r"^(hm\d+)-(\d{2})-(\d{2})-(\d{4})\.txt$")
    pattern = re.compile(r"(\d{2})-(\d{2})-(\d{4})\.txt$")
    
    # Get the list of files in the current directory
    logs_path = "./logs"
    files = os.listdir(logs_path)
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            day, month, year = match.groups()
            new_filename = f"hm40-{year}-{month}-{day}.txt"
            os.rename(logs_path+'/'+filename, logs_path+'/'+new_filename)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_files()
