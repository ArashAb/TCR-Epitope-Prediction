import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from .utils import get_esm2_embeddings

def load_data(file_path, chunk_size=10000):
    """
    Load data from CSV file in chunks.
    
    Args:
    file_path (str): Path to the CSV file
    chunk_size (int): Number of rows to read at a time
    
    Returns:
    pd.DataFrame: Iterator of data chunks
    dict: Label encoders for categorical columns
    LabelEncoder: Encoder for epitope (target) column
    """
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Initialize label encoders
    label_encoders = {}
    for col in ['TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'MHC', 'MHC_class', 'species']:
        label_encoders[col] = LabelEncoder()
    
    epitope_encoder = LabelEncoder()
    
    return chunks, label_encoders, epitope_encoder

def preprocess_data(data, label_encoders, epitope_encoder):
    """
    Preprocess a chunk of data.
    
    Args:
    data (pd.DataFrame): Data chunk to preprocess
    label_encoders (dict): Label encoders for categorical columns
    epitope_encoder (LabelEncoder): Encoder for epitope (target) column
    
    Returns:
    pd.DataFrame: Preprocessed data
    """
    for col, encoder in label_encoders.items():
        data[f'{col}_encoded'] = encoder.fit_transform(data[col])
    
    data['epitope_encoded'] = epitope_encoder.fit_transform(data['epitope'])
    
    # Get ESM2 embeddings for CDR3_TRA and CDR3_TRB
    data['cdr3_TRA_embedding'] = list(get_esm2_embeddings(data['cdr3_TRA']))
    data['cdr3_TRB_embedding'] = list(get_esm2_embeddings(data['cdr3_TRB']))
    
    return data

class TCRDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'cdr3_TRA': torch.tensor(item['cdr3_TRA_embedding'], dtype=torch.float32),
            'cdr3_TRB': torch.tensor(item['cdr3_TRB_embedding'], dtype=torch.float32),
            'TRAV': torch.tensor(item['TRAV_encoded'], dtype=torch.long),
            'TRAJ': torch.tensor(item['TRAJ_encoded'], dtype=torch.long),
            'TRBV': torch.tensor(item['TRBV_encoded'], dtype=torch.long),
            'TRBJ': torch.tensor(item['TRBJ_encoded'], dtype=torch.long),
            'MHC': torch.tensor(item['MHC_encoded'], dtype=torch.long),
            'MHC_class': torch.tensor(item['MHC_class_encoded'], dtype=torch.long),
            'species': torch.tensor(item['species_encoded'], dtype=torch.long),
            'epitope': torch.tensor(item['epitope_encoded'], dtype=torch.long)
        }
