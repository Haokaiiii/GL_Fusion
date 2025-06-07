"""
Graph Neural Network implementations for the GL-Fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data


class GAT(nn.Module):
    """Graph Attention Network (GAT) model."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            # For single layer, don't concatenate heads to maintain output dimension
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=False))
        else:
            # Input layer
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
                
            # Output layer - don't concatenate to get correct output dimension
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GAT.
        Applies GATConv -> ReLU -> Dropout for each layer.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # No activation/dropout on the final layer's output
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) model for learning node embeddings.
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.2):
        """
        Initialize the GCN model.
        
        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension.
            out_dim (int): Output dimension.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, out_dim))
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, in_dim].
            edge_index (torch.Tensor): Graph adjacency information [2, num_edges].
            
        Returns:
            torch.Tensor: Updated node embeddings [num_nodes, out_dim].
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        
        return x


def create_gnn_model(model_config, num_node_features):
    """
    Factory function to create a GNN model based on configuration.
    
    Args:
        model_config (dict): GNN configuration parameters.
        num_node_features (int): Number of input node features.
        
    Returns:
        nn.Module: GNN model instance.
    """
    model_type = model_config.get('model_type', 'GAT')
    hidden_dim = model_config.get('hidden_dim', 128)
    out_dim = model_config.get('hidden_dim', 128)  # Default to same as hidden_dim
    num_layers = model_config.get('num_layers', 3)
    dropout = model_config.get('dropout', 0.2)
    
    if model_type == 'GAT':
        heads = model_config.get('heads', 4)
        return GAT(
            input_dim=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
    elif model_type == 'GCN':
        return GCN(
            in_dim=num_node_features,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported GNN model type: {model_type}") 