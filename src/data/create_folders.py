import os

def create_folder_structure():
    # Define the base directories
    data_dirs = [
        'data/IXI_T1',
        'data/IXI_T2',
        'data/trainA',
        'data/trainB',
        'data/testA',
        'data/testB',
        'data/valA',
        'data/valB'
    ]
    
    # Create directories if they don't exist
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    create_folder_structure() 