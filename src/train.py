import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .data_processing import preprocess_data, TCRDataset
from .utils import evaluate

def train_model(model, data, label_encoders, epitope_encoder, file_path, batch_size=32, epochs=10, learning_rate=1e-4, val_split=0.2):
    """
    Train the TCR Classifier model.
    
    Args:
    model (nn.Module): The model to train
    data (pd.DataFrame): Iterator of data chunks
    label_encoders (dict): Label encoders for categorical columns
    epitope_encoder (LabelEncoder): Encoder for epitope (target) column
    batch_size (int): Batch size for training
    epochs (int): Number of epochs to train
    learning_rate (float): Learning rate for optimizer
    val_split (float): Proportion of data to use for validation
    
    Returns:
    nn.Module: The trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_loss_list = []
        val_loss_list = []
        model.train()
        
        for chunk in data:
            preprocessed_data = preprocess_data(chunk, label_encoders, epitope_encoder)
            train_data, val_data = train_test_split(preprocessed_data, test_size=val_split, random_state=42)
            
            train_dataset = TCRDataset(train_data)
            val_dataset = TCRDataset(val_data)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training loop
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch['epitope'])
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
            
            # Validation
            val_loss = evaluate(model, val_dataloader, criterion)
            val_loss_list.append(val_loss)
        
        # Print epoch results
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model
