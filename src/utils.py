import torch
import esm

# Load ESM-2 model
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.eval()

def get_esm2_embeddings(sequences):
    """
    Generate ESM-2 embeddings for a list of sequences.
    
    Args:
    sequences (list): List of protein sequences
    
    Returns:
    numpy.ndarray: Array of sequence embeddings
    """
    sequences = [(f"prot{i}", seq) for i, seq in enumerate(sequences)]

    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    
    return torch.stack(sequence_representations).numpy()

def evaluate(model, dataloader, criterion):
    """
    Evaluate the model on a given dataloader.
    
    Args:
    model (nn.Module): The model to evaluate
    dataloader (DataLoader): The dataloader containing the evaluation data
    criterion (nn.Module): The loss function
    
    Returns:
    float: Average loss on the evaluation data
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            loss = criterion(outputs, batch['epitope'])
            total_loss += loss.item()
    return total_loss / len(dataloader)
