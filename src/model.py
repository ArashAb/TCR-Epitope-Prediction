import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TCRClassifier(nn.Module):
    def __init__(self, num_classes, vocab_sizes, embed_dim=128, num_heads=8, num_layers=3):
        super(TCRClassifier, self).__init__()
        self.cdr3_proj = nn.Linear(1280, embed_dim)  # ESM2 output dimension is 1280
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embed_dim)
            for col, vocab_size in vocab_sizes.items()
        })
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Process CDR3 sequences
        cdr3_tra = self.cdr3_proj(x['cdr3_TRA'])
        cdr3_trb = self.cdr3_proj(x['cdr3_TRB'])
        
        # Process other features
        other_features = [self.embeddings[col](x[col]) for col in self.embeddings.keys()]
        
        # Combine all features
        combined = torch.stack([cdr3_tra, cdr3_trb] + other_features, dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_encoder:
            combined = layer(combined)
        
        # Global average pooling
        pooled = combined.mean(dim=1)
        
        # Classification layer
        output = self.fc(pooled)
        return output
